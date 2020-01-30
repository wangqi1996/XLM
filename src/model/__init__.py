# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
from logging import getLogger
import os
import torch

from src.model.KD import KD_encoder
from src.model.add_pretrain import AddPretrainTransModel
from src.model.attn_transformers import PretrainAttnTransModel
from src.model.elmo_embedder import ELMOTransEncoder
from src.model.fusion_transformer import FusionTransEncoder
from .pretrain import load_embeddings
from .transformer import DECODER_ONLY_PARAMS, TransformerModel  # , TRANSFORMER_LAYER_PARAMS
from .memory import HashingMemory

logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # masked language modeling task parameters
    assert params.bptt >= 1
    assert 0 <= params.word_pred < 1
    assert 0 <= params.sample_alpha < 1
    s = params.word_mask_keep_rand.split(',')
    assert len(s) == 3
    s = [float(x) for x in s]
    assert all([0 <= x <= 1 for x in s]) and sum(s) == 1
    params.word_mask = s[0]
    params.word_keep = s[1]
    params.word_rand = s[2]

    # input sentence noise for DAE
    if len(params.ae_steps) == 0:
        assert params.word_shuffle == 0
        assert params.word_dropout == 0
        assert params.word_blank == 0
    else:
        assert params.word_shuffle == 0 or params.word_shuffle > 1
        assert 0 <= params.word_dropout < 1
        assert 0 <= params.word_blank < 1

    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # share input and output embeddings
    assert params.share_inout_emb is False or params.asm is False

    # adaptive softmax
    if params.asm:
        assert params.asm_div_value > 1
        s = params.asm_cutoffs.split(',')
        assert all([x.isdigit() for x in s])
        params.asm_cutoffs = [int(x) for x in s]
        assert params.max_vocab == -1 or params.asm_cutoffs[-1] < params.max_vocab

    # memory
    if params.use_memory:
        HashingMemory.check_params(params)
        s_enc = [x for x in params.mem_enc_positions.split(',') if x != '']
        s_dec = [x for x in params.mem_dec_positions.split(',') if x != '']
        assert len(s_enc) == len(set(s_enc))
        assert len(s_dec) == len(set(s_dec))
        assert all(x.isdigit() or x[-1] == '+' and x[:-1].isdigit() for x in s_enc)
        assert all(x.isdigit() or x[-1] == '+' and x[:-1].isdigit() for x in s_dec)
        params.mem_enc_positions = [(int(x[:-1]), 'after') if x[-1] == '+' else (int(x), 'in') for x in s_enc]
        params.mem_dec_positions = [(int(x[:-1]), 'after') if x[-1] == '+' else (int(x), 'in') for x in s_dec]
        assert len(params.mem_enc_positions) + len(params.mem_dec_positions) > 0
        assert len(params.mem_enc_positions) == 0 or 0 <= min([x[0] for x in params.mem_enc_positions]) <= max(
            [x[0] for x in params.mem_enc_positions]) <= params.n_layers - 1
        assert len(params.mem_dec_positions) == 0 or 0 <= min([x[0] for x in params.mem_dec_positions]) <= max(
            [x[0] for x in params.mem_dec_positions]) <= params.n_layers - 1

    # reload pretrained word embeddings
    if params.reload_emb != '':
        assert os.path.isfile(params.reload_emb)

    if params.reload_emb_from_xml != '':
        assert os.path.isfile(params.reload_emb_from_xml)

    # reload a pretrained model
    if params.reload_model != '':
        if params.encoder_only:
            assert os.path.isfile(params.reload_model)
        else:
            s = params.reload_model.split(',')
            assert len(s) == 2
            assert all([x == '' or os.path.isfile(x) for x in s])


def set_pretrain_emb(model, dico, word2id, embeddings):
    """
    Pretrain word embeddings.
    """
    n_found = 0
    with torch.no_grad():
        for i in range(len(dico)):
            idx = word2id.get(dico[i], None)
            if idx is None:
                continue
            n_found += 1
            model.embeddings.weight[i] = embeddings[idx].cuda()
            model.pred_layer.proj.weight[i] = embeddings[idx].cuda()
    logger.info("Pretrained %i/%i words (%.3f%%)."
                % (n_found, len(dico), 100. * n_found / len(dico)))


def build_model(params, dico):
    """
    Build model.
    """
    if params.encoder_only:
        # build
        model = TransformerModel(params, dico, is_encoder=True, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(model, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            logger.info("Reloading model from %s ..." % params.reload_model)
            reloaded = \
                torch.load(params.reload_model, map_location=lambda storage, loc: storage.cuda(params.local_rank))[
                    'model']
            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            # # HACK to reload models with less layers
            # for i in range(12, 24):
            #     for k in TRANSFORMER_LAYER_PARAMS:
            #         k = k % i
            #         if k in model.state_dict() and k not in reloaded:
            #             logger.warning("Parameter %s not found. Ignoring ..." % k)
            #             reloaded[k] = model.state_dict()[k]

            model.load_state_dict(reloaded)

        logger.info("Model: {}".format(model))
        logger.info(
            "Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))

        return model.cuda()

    else:
        # 使用language model的输出作为翻译模型的输入
        if params.encoder_elmo_path != '':
            # 在这里加载一个预训练好的模型
            logger.info("Reloading encoder_elmo_path from %s ..." % params.encoder_elmo_path)
            pretrain_model = TransformerModel(params, dico, is_encoder=True, with_output=True)
            reloaded = \
                torch.load(params.encoder_elmo_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))[
                    'model']
            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            pretrain_model.load_state_dict(reloaded)
            params.language_model = pretrain_model
            encoder = ELMOTransEncoder(params, dico, is_encoder=True, with_output=True)
        elif params.encoder_fusion_path != '':
            # 按照wengrx师兄的论文实现train
            logger.info("Reloading encoder_fusion_path from %s ..." % params.encoder_fusion_path)
            pretrain_model = TransformerModel(params, dico, is_encoder=True, with_output=True)
            reloaded = \
                torch.load(params.encoder_fusion_path,
                           map_location=lambda storage, loc: storage.cuda(params.local_rank))[
                    'model']
            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            pretrain_model.load_state_dict(reloaded)
            params.fusion_model = pretrain_model
            encoder = FusionTransEncoder(params, dico, is_encoder=True, with_output=True, style=params.fusion_style)
        elif params.pretrain_attn_model_path != '':
            # 按照wengrx师兄的论文实现train
            logger.info("Reloading pretrain_attn_model_path from %s ..." % params.pretrain_attn_model_path)
            pretrain_model = TransformerModel(params, dico, is_encoder=True, with_output=True)
            reloaded = \
                torch.load(params.pretrain_attn_model_path,
                           map_location=lambda storage, loc: storage.cuda(params.local_rank))[
                    'model']
            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            pretrain_model.load_state_dict(reloaded)
            params.pretrain_model = pretrain_model
            encoder = PretrainAttnTransModel(params, dico, is_encoder=True, with_output=True)
        elif params.add_pretrain_path != '':
            # 按照审稿论文实现
            logger.info("Reloading add_pretrain_path from %s ..." % params.add_pretrain_path)
            pretrain_model = TransformerModel(params, dico, is_encoder=True, with_output=True)
            reloaded = \
                torch.load(params.add_pretrain_path,
                           map_location=lambda storage, loc: storage.cuda(params.local_rank))[
                    'model']
            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            pretrain_model.load_state_dict(reloaded)
            params.pretrain_model = pretrain_model
            encoder = AddPretrainTransModel(params, dico, is_encoder=True, with_output=True)
        else:
            # build
            encoder = TransformerModel(params, dico, is_encoder=True,
                                       with_output=True)  # TODO: only output when necessary - len(params.clm_steps + params.mlm_steps) > 0
        decoder = TransformerModel(params, dico, is_encoder=False, with_output=True)

        # encoder端使用知识蒸馏
        if params.encoder_KD != '':
            logger.info("Reloading encoder KD from %s ..." % params.encoder_KD)
            pretrain_model = TransformerModel(params, dico, is_encoder=True, with_output=True)
            reloaded = \
                torch.load(params.encoder_KD,
                           map_location=lambda storage, loc: storage.cuda(params.local_rank))[
                    'model']
            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            pretrain_model.load_state_dict(reloaded)
            params.encoder_KD_model = pretrain_model.cuda()
            params.KD_encoder = KD_encoder(params)
        else:
            params.encoder_KD_model = None

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(encoder, dico, word2id, embeddings)
            set_pretrain_emb(decoder, dico, word2id, embeddings)

        # 使用xlm的embedding来初始化
        if params.reload_emb_from_xml != '':
            emb_path = params.reload_emb_from_xml

            if emb_path != '':
                logger.info("Reloading embedding from %s ..." % emb_path)
                emb_reload = torch.load(emb_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                emb_reload = emb_reload['model' if 'model' in emb_reload else 'encoder']
                emb_name = ['position_embeddings', 'lang_embeddings', 'embeddings']

                if all([k.startswith('module.') for k in emb_reload.keys()]):
                    emb_reload = {k[len('module.'):]: v for k, v in emb_reload.items()}

                only_emb_reload = {}
                for key, value in emb_reload.items():
                    for module_name in emb_name:
                        if module_name in key:
                            only_emb_reload[key] = value

                # 预训练模型只有一个，所以使用相同的embedding初始化就可以
                enc_reload = encoder.state_dict()
                enc_reload.update(only_emb_reload)
                encoder.load_state_dict(enc_reload)

                dec_reload = decoder.state_dict()
                dec_reload.update(only_emb_reload)
                decoder.load_state_dict(dec_reload)

        # 使用xlm的embedding来初始化
        if params.reload_dec_emb_from_xml != '':
            emb_path = params.reload_dec_emb_from_xml

            if emb_path != '':
                logger.info("Reloading decoder embedding from %s ..." % emb_path)
                emb_reload = torch.load(emb_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                emb_reload = emb_reload['model' if 'model' in emb_reload else 'encoder']
                emb_name = ['position_embeddings', 'lang_embeddings', 'embeddings']

                if all([k.startswith('module.') for k in emb_reload.keys()]):
                    emb_reload = {k[len('module.'):]: v for k, v in emb_reload.items()}

                only_emb_reload = {}
                for key, value in emb_reload.items():
                    for module_name in emb_name:
                        if module_name in key:
                            only_emb_reload[key] = value

                dec_reload = decoder.state_dict()
                dec_reload.update(only_emb_reload)
                decoder.load_state_dict(dec_reload)

        # reload a pretrained model
        if params.reload_model != '':
            enc_path, dec_path = params.reload_model.split(',')
            assert not (enc_path == '' and dec_path == '')

            # reload encoder
            if enc_path != '':
                logger.info("Reloading encoder from %s ..." % enc_path)
                enc_reload = torch.load(enc_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                enc_reload = enc_reload['model' if 'model' in enc_reload else 'encoder']
                if all([k.startswith('module.') for k in enc_reload.keys()]):
                    enc_reload = {k[len('module.'):]: v for k, v in enc_reload.items()}
                encoder.load_state_dict(enc_reload)

            # reload decoder
            if dec_path != '':
                logger.info("Reloading decoder from %s ..." % dec_path)
                dec_reload = torch.load(dec_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                dec_reload = dec_reload['model' if 'model' in dec_reload else 'decoder']
                if all([k.startswith('module.') for k in dec_reload.keys()]):
                    dec_reload = {k[len('module.'):]: v for k, v in dec_reload.items()}
                for i in range(params.n_layers):
                    for name in DECODER_ONLY_PARAMS:
                        if name % i not in dec_reload:
                            logger.warning("Parameter %s not found." % (name % i))
                            dec_reload[name % i] = decoder.state_dict()[name % i]
                decoder.load_state_dict(dec_reload)

        # reload a pretrained model for encoder
        if params.reload_encoder_model != '':
            enc_path = params.reload_encoder_model

            # reload encoder
            if enc_path != '':
                logger.info("Reloading encoder pretrain model from %s ..." % enc_path)
                enc_reload = torch.load(enc_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                enc_reload = enc_reload['model' if 'model' in enc_reload else 'encoder']
                if all([k.startswith('module.') for k in enc_reload.keys()]):
                    enc_reload = {k[len('module.'):]: v for k, v in enc_reload.items()}
                encoder.load_state_dict(enc_reload)

        # embedding层不微调
        if params.froze_enc_embedding:
            logger.info("froze_enc_embedding")
            emb_name = ['position_embeddings', 'lang_embeddings', 'embeddings', 'layer_norm_emb']
            for name, param in encoder.named_parameters():
                for _name in emb_name:
                    if _name in name:
                        param.requires_grad = False
                        break

        # encoder的某几层不微调
        if params.froze_enc:
            logger.info("froze encoder")
            layer = params.froze_layer  # 3
            emb_name = ['position_embeddings', 'lang_embeddings', 'embeddings', 'layer_norm_emb']
            layer_name = ['attentions.', 'layer_norm1.', 'ffns.', 'layer_norm2.']
            new_layer_name = copy.deepcopy(emb_name)
            for i in range(layer):
                new_layer_name.extend([_name + str(i) for _name in layer_name])

            for name, param in encoder.named_parameters():
                for _name in new_layer_name:
                    if _name in name:
                        param.requires_grad = False
                        break

        logger.debug("Encoder: {}".format(encoder))
        logger.debug("Decoder: {}".format(decoder))
        logger.info(
            "Number of parameters (encoder): %i" % sum([p.numel() for p in encoder.parameters() if p.requires_grad]))
        logger.info(
            "Number of parameters (decoder): %i" % sum([p.numel() for p in decoder.parameters() if p.requires_grad]))

        return encoder.cuda(), decoder.cuda()
