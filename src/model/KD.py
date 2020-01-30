# coding=utf-8
import torch
from torch import nn


class KD_encoder(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.KD_model = params.encoder_KD_model
        self.alpha = nn.Parameter(torch.FloatTensor(1).fill_(1).float(), requires_grad=True).cuda()

    def forward(self, loss, encoder_hiddens, x1, len1, langs1):
        with torch.no_grad():
            if self.KD_model.training:
                self.KD_model.eval()
            _, teacher_hiddens = self.KD_model('fwd', x=x1, lengths=len1, langs=langs1, causal=False,
                                               output_hidden=True)
        # [num_layers, batch_size, seq_len, hidden_dim]
        encoder_hiddens = torch.stack(encoder_hiddens, dim=0)
        teacher_hiddens = torch.stack(teacher_hiddens, dim=0)

        diff = (encoder_hiddens - teacher_hiddens) * (encoder_hiddens - teacher_hiddens)
        # [batch_size, hidden_dim]
        diff = diff.sum(dim=-2, keepdim=False).sum(dim=0, keepdim=False)
        diff = diff / (len1.unsqueeze(-1).float())
        # loss是一维的, 我貌似只能这样拼了
        loss2 = diff.mean(dim=-1, keepdim=False).mean(dim=-1, keepdim=False)

        loss = loss.reshape(1) + loss2 * self.alpha

        return loss
