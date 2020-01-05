

处理数据
./get-data-nmt.sh --src en --tgt ro --reload_codes /home/user_data55/wangdq/model/XLM/data/codes_enro --reload_vocab /home/user_data55/wangdq/model/XLM/data/vocab_enro

监督机器翻译  ro-en

python train.py

## main parameters
--exp_name supMT_enro                                       # experiment name
--dump_path /home/user_data55/wangdq/model/XLM/enro                                       # where to store the experiment
--reload_model '/home/user_data55/wangdq/model/XLM/mlm_enro_1024.pth,/home/user_data55/wangdq/model/XLM/mlm_enro_1024.pth'          # model to reload for encoder,decoder

## data location / training objective
--data_path ./data/processed/en-ro/                           # data location
--lgs 'ro-en'                                                 # considered languages
--mt_steps 'ro-en'                                            # denoising auto-encoder training steps
--word_shuffle 0                                              # noise for auto-encoding loss
--word_dropout 0                                            # noise for auto-encoding loss
--word_blank 0                                             # noise for auto-encoding loss
--lambda_mt '0:1,100000:0.1,300000:0'                         # scheduling on the auto-encoding coefficient

## transformer parameters
--encoder_only false                                          # use a decoder for MT
--emb_dim 1024                                                # embeddings / model dimension
--n_layers 6                                                  # number of layers
--n_heads 8                                                   # number of heads
--dropout 0.1                                                 # dropout
--attention_dropout 0.1                                       # attention dropout
--gelu_activation true                                        # GELU instead of ReLU

## optimization
--tokens_per_batch 2000                                       # use batches with a fixed number of words
--batch_size 32                                               # batch size (for back-translation)
--bptt 256                                                    # sequence length
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001  # optimizer
--epoch_size 200000                                           # number of sentences per epoch
--eval_bleu true                                              # also evaluate the BLEU score
--stopping_criterion 'valid_ro-en_mt_bleu,10'                 # validation metric (when to save the best model)
--validation_metrics 'valid_ro-en_mt_bleu'                    # end experiment if stopping criterion does not improve