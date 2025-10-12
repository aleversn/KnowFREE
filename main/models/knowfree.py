import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_max
import numpy as np
from transformers import BertModel, BertPreTrainedModel, BertConfig
from fastNLP import seq_len_to_mask
from typing import Tuple


class KnowFREE(BertPreTrainedModel):
    def __init__(self, config: BertConfig):

        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.size_embed_dim = config.size_embed_dim
        self.cnn_dim = config.cnn_dim
        self.biaffine_size = config.biaffine_size
        self.logit_drop = config.logit_drop
        self.kernel_size = config.kernel_size
        self.n_head = config.n_head
        self.cnn_depth = config.cnn_depth
        self.num_labels = config.num_labels
        self.span_threshold = config.span_threshold
        self.ext_labels_start_idx = config.num_target_labels

        self.bert = BertModel(config, add_pooling_layer=False)

        if self.size_embed_dim != 0:
            n_pos = 30
            self.size_embedding = torch.nn.Embedding(
                n_pos, self.size_embed_dim)

            _span_size_ids = torch.arange(
                512) - torch.arange(512).unsqueeze(-1)

            _span_size_ids.masked_fill_(_span_size_ids < -n_pos/2, -n_pos/2)
            _span_size_ids = _span_size_ids.masked_fill(
                _span_size_ids >= n_pos/2, n_pos/2-1) + n_pos/2

            self.register_buffer('span_size_ids', _span_size_ids.long())
            hsz = self.biaffine_size*2 + self.size_embed_dim + 2
        else:
            hsz = self.biaffine_size*2+2
        biaffine_input_size = self.hidden_size

        self.head_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(biaffine_input_size, self.biaffine_size),
            nn.LeakyReLU(),
        )
        self.tail_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(biaffine_input_size, self.biaffine_size),
            nn.LeakyReLU(),
        )

        self.dropout = nn.Dropout(0.4)
        if self.n_head > 0:
            self.multi_head_biaffine = MultiHeadBiaffine(
                self.biaffine_size, self.cnn_dim, n_head=self.n_head)
        else:
            self.U = nn.Parameter(torch.randn(
                self.cnn_dim, self.biaffine_size, self.biaffine_size))
            torch.nn.init.xavier_normal_(self.U.data)
        self.W = torch.nn.Parameter(torch.empty(self.cnn_dim, hsz))
        torch.nn.init.xavier_normal_(self.W.data)
        if self.cnn_depth > 0:
            self.cnn = MaskCNN(self.cnn_dim, self.cnn_dim,
                               kernel_size=self.kernel_size, depth=self.cnn_depth)
        self.attn = LocalAttentionModel(self.cnn_dim, self.kernel_size)

        self.down_fc = nn.Linear(self.cnn_dim, self.num_labels-1)
        self.logit_drop = self.logit_drop

    def decode_labels(self, labels: torch.Tensor, indexes: torch.Tensor):

        length: np.ndarray = indexes.detach().cpu().numpy()
        length = length.max(-1)
        labels[:, :, :, self.ext_labels_start_idx:] = 0
        labels: np.ndarray = labels.detach().cpu().numpy()
        span_mask = (labels.max(-1) > self.span_threshold)
        labels = labels.argmax(-1)
        indexes = np.where(span_mask)
        entities = [set() for _ in range(labels.shape[0])]
        for batch, x, y in zip(*indexes):
            if x <= y and x >= 0 and y >= 0 and x < length[batch] and y < length[batch]:
                entities[batch].add(
                    (x, y, labels[batch, x, y] + 1))
        return entities

    def is_span_intersect(self, a: Tuple[int, int], b: Tuple[int, int]):
        """
            Determine whether two intervals intersect, with both the left and right intervals being closed
        """
        return a[0] <= b[1] and b[0] <= a[1]

    def is_span_nested(self, a: Tuple[int, int], b: Tuple[int, int]):
        """
            Determine whether two intervals are nested, with both the left and right intervals being closed
        """
        return (b[0] <= a[0] and a[1] <= b[1]) or (a[0] <= b[0] and b[1] <= a[1])

    def decode_logits(self, scores: torch.Tensor, indexes: torch.Tensor, remove_clashed: bool = False, nested: bool = True):
        scores = scores.sigmoid()

        scores: np.ndarray = scores.detach().cpu().numpy()

        length: np.ndarray = indexes.detach().cpu().numpy()
        length = length.max(-1)

        scores[:, :, :, self.ext_labels_start_idx:] = 0
        span_mask = (scores.max(-1) > self.span_threshold)
        argmax = scores.argmax(-1)
        indexes = np.where(span_mask)
        entities = [[] for _ in range(scores.shape[0])]

        for batch_idx, x, y in zip(*indexes):
            if x >= 0 and x < length[batch_idx] and y >= 0 and y < length[batch_idx] and x <= y:

                entities[batch_idx].append(
                    (x, y, argmax[batch_idx, x, y] + 1, scores[batch_idx, x, y, argmax[batch_idx, x, y]]))

        for batch_idx in range(len(entities)):
            entities[batch_idx].sort(key=lambda x: x[-1], reverse=True)
        if remove_clashed:
            for batch_idx in range(len(entities)):
                new_entities = []
                for entity in entities[batch_idx]:
                    add = True
                    for pre_entity in new_entities:
                        if self.is_span_intersect(entity, pre_entity) and (not nested or not self.is_span_nested(entity, pre_entity)):
                            add = False
                            break
                    if add:
                        new_entities.append(entity)
                entities[batch_idx] = new_entities

        for batch_idx in range(len(entities)):
            entities[batch_idx] = set(
                map(lambda x: (x[0], x[1], x[2]), entities[batch_idx]))
        return entities

    def forward(self, input_ids: torch.Tensor, bpe_len: torch.Tensor, indexes: torch.Tensor, labels: torch.Tensor = None, label_weights: torch.Tensor = None, is_synthetic: torch.Tensor = None, **kwargs):

        attention_mask = seq_len_to_mask(bpe_len)
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_states = outputs['last_hidden_state']

        state = scatter_max(last_hidden_states, index=indexes, dim=1)[
            0][:, 1:]

        lengths, _ = indexes.max(dim=-1)

        head_state = self.head_mlp(state)
        tail_state = self.tail_mlp(state)

        if hasattr(self, 'U'):
            scores1 = torch.einsum(
                'bxi, oij, byj -> boxy', head_state, self.U, tail_state)
        else:

            scores1 = self.multi_head_biaffine(head_state, tail_state)

        head_state = torch.cat(
            [head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat(
            [tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        affined_cat = torch.cat([self.dropout(head_state).unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                                 self.dropout(tail_state).unsqueeze(1).expand(-1, head_state.size(1), -1, -1)], dim=-1)

        if hasattr(self, 'size_embedding'):
            size_embedded = self.size_embedding(
                self.span_size_ids[:state.size(1), :state.size(1)])
            affined_cat = torch.cat([affined_cat,
                                     self.dropout(size_embedded).unsqueeze(0).expand(state.size(0), -1, -1, -1)], dim=-1)

        scores2 = torch.einsum('bmnh,kh->bkmn', affined_cat,
                               self.W)
        scores = scores2 + scores1

        if hasattr(self, 'cnn'):
            mask = seq_len_to_mask(lengths)
            mask = mask[:, None] * mask.unsqueeze(-1)
            pad_mask = mask[:, None].eq(0)
            u_scores = scores.masked_fill(pad_mask, 0)
            if self.logit_drop != 0:
                u_scores = F.dropout(
                    u_scores, p=self.logit_drop, training=self.training)

            u_scores = self.attn(u_scores.permute(
                0, 2, 3, 1), pad_mask=pad_mask.permute(0, 2, 3, 1))
            scores = u_scores.permute(0, 3, 1, 2) + scores

        scores = self.down_fc(scores.permute(0, 2, 3, 1))

        assert scores.size(-1) == labels.size(-1)

        loss = None
        if labels is not None:
            if label_weights is None:
                label_weights = 0.13
            flat_scores = scores.reshape(-1)
            flat_matrix = labels.reshape(-1)
            decay_weights = torch.ones(labels.size()).to(flat_matrix.device)
            decay_weights *= label_weights
            decayed_weights = decay_weights.reshape(input_ids.size(0), -1)
            synthetic_mask = torch.ones(labels.size()).to(flat_matrix.device)
            synthetic_mask[:, is_synthetic] *= 1
            synthetic_weights = synthetic_mask.reshape(input_ids.size(0), -1)
            mask = flat_matrix.ne(-100).float().view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(
                flat_scores, flat_matrix.float(), reduction='none')
            loss = ((flat_loss.view(input_ids.size(0), -1) *
                    synthetic_weights*decayed_weights*mask).sum(dim=-1)).mean()

        return loss, scores


class LocalSpanAttention(nn.Module):
    def __init__(self, dim):
        super(LocalSpanAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=10)

    def forward(self, x, span_mask):
        """
        :param x: [bsz, len, len, dim] Input feature matrix
        :param span_mask: [bsz, len, len] The mask matrix is used to control the receptive field of attention
        """
        bsz, length, _, dim = x.shape

        x = x.view(bsz * length, length, dim)

        x = x.transpose(0, 1)

        attention_output, _ = self.attn(x, x, x, attn_mask=span_mask)

        attention_output = attention_output.transpose(
            0, 1).view(bsz, length, length, dim)

        return attention_output


class LocalAttentionModel(nn.Module):
    def __init__(self, dim, window_size=3):
        super(LocalAttentionModel, self).__init__()
        self.local_attention = LocalSpanAttention(dim)
        self.norm = nn.LayerNorm(dim)
        self.window_size = window_size

    def generate_local_mask(self, seq_len, window_size):

        mask = torch.full((seq_len, seq_len), float('-inf'))
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            mask[i, start:end] = 0
        return mask

    def forward(self, x, pad_mask):
        """
        :param x: [bsz, len, len, dim] Input feature matrix
        """
        bsz, length, _, dim = x.shape

        local_mask = self.generate_local_mask(length, self.window_size)
        local_mask = local_mask.to(x.device)

        x = x.masked_fill(pad_mask, 0)
        attn_output = self.local_attention(x, local_mask)

        output = self.norm(attn_output)

        return output


class LayerNorm(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim_index=1):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim_index = dim_index
        self.eps = 1e-6

    def forward(self, x):
        """

        :param x: bsz x dim x max_len x max_len
        :param mask: bsz x dim x max_len x max_len, The place where it is 1 is pad
        :return:
        """
        u = x.mean(dim=self.dim_index, keepdim=True)
        s = (x - u).pow(2).mean(dim=self.dim_index, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x


class MaskConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, groups=1):
        super(MaskConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding,
                                bias=False, groups=groups)

    def forward(self, x, mask):
        """

        :param x:
        :param mask:
        :return:
        """
        x = x.masked_fill(mask, 0)
        _x = self.conv2d(x)
        return _x


class MaskCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, depth=3):
        super(MaskCNN, self).__init__()

        layers = []
        for i in range(depth):
            layers.extend([
                MaskConv2d(input_channels, input_channels,
                           kernel_size=kernel_size, padding=kernel_size//2),
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU()])
        layers.append(MaskConv2d(input_channels, output_channels,
                      kernel_size=3, padding=3//2))
        self.cnns = nn.ModuleList(layers)

    def forward(self, x, mask):
        _x = x
        for layer in self.cnns:
            if isinstance(layer, LayerNorm):
                x = x + _x
                x = layer(x)
                _x = x
            elif not isinstance(layer, nn.GELU):
                x = layer(x, mask)
            else:
                x = layer(x)
        return _x


class MultiHeadBiaffine(nn.Module):
    def __init__(self, dim, out=None, n_head=4):
        super(MultiHeadBiaffine, self).__init__()
        assert dim % n_head == 0
        in_head_dim = dim//n_head
        out = dim if out is None else out
        assert out % n_head == 0
        out_head_dim = out//n_head
        self.n_head = n_head
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.randn(
            self.n_head, out_head_dim, in_head_dim, in_head_dim)))
        self.out_dim = out

    def forward(self, h, v):
        """

        :param h: bsz x max_len x dim
        :param v: bsz x max_len x dim
        :return: bsz x max_len x max_len x out_dim
        """
        bsz, max_len, dim = h.size()
        h = h.reshape(bsz, max_len, self.n_head, -1)
        v = v.reshape(bsz, max_len, self.n_head, -1)

        w = torch.einsum('blhx,hdxy,bkhy->bhdlk', h, self.W, v)

        w = w.reshape(bsz, self.out_dim, max_len, max_len)
        return w
