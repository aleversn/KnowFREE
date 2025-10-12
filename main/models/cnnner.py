import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_max
import numpy as np
from transformers import BertModel, BertPreTrainedModel, BertConfig
from fastNLP import seq_len_to_mask
from typing import Tuple


class CNNNerv1(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        #  model_name, num_ner_tag, cnn_dim=200, biaffine_size=200,
        #  size_embed_dim=0, logit_drop=0, kernel_size=3, n_head=4, cnn_depth=3):
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

        self.bert = BertModel(config, add_pooling_layer=False)

        if self.size_embed_dim != 0:
            n_pos = 30 # span 跨度位置编码为-n_pos到n_pos之间
            self.size_embedding = torch.nn.Embedding(
                n_pos, self.size_embed_dim)
            # `512 - 512`: 这两个生成的张量相减，得到一个矩阵，每个元素代表两个位置之间的距离（跨度）。 
            # e.g. [[0,1,2,...,512]
            # [-1,0,1,...,511]
            # [...]
            # [-511,-510,...,0]]
            _span_size_ids = torch.arange(
                512) - torch.arange(512).unsqueeze(-1)
            # 限制span最大距离为pos / 2
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos/2, -n_pos/2)
            _span_size_ids = _span_size_ids.masked_fill(
                _span_size_ids >= n_pos/2, n_pos/2-1) + n_pos/2
            # 注册为非更新参数
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

        self.down_fc = nn.Linear(self.cnn_dim, self.num_labels-1)
        self.logit_drop = self.logit_drop

    def decode_labels(self, labels: torch.Tensor, indexes: torch.Tensor):
        # 这里的labels不含有特殊的字符，因此不需要减去offset
        length: np.ndarray = indexes.detach().cpu().numpy()
        length = length.max(-1)
        labels: np.ndarray = labels.detach().cpu().numpy()
        span_mask = (labels.max(-1) > self.span_threshold)
        labels = labels.argmax(-1)
        indexes = np.where(span_mask)
        entities = [set() for _ in range(labels.shape[0])]
        for batch, x, y in zip(*indexes):
            if x <= y and x >= 0 and y >= 0 and x < length[batch] and y < length[batch]:
                entities[batch].add(
                    (x, y, labels[batch, x, y] + 1))  # +1 是由于有O标签
        return entities

    def is_span_intersect(self, a: Tuple[int, int], b: Tuple[int, int]):
        """
            判断两个区间是否相交，左右都是闭区间
        """
        return a[0] <= b[1] and b[0] <= a[1]

    def is_span_nested(self, a: Tuple[int, int], b: Tuple[int, int]):
        """
            判断两个区间是否嵌套，左右都是闭区间
        """
        return (b[0] <= a[0] and a[1] <= b[1]) or (a[0] <= b[0] and b[1] <= a[1])

    def decode_logits(self, scores: torch.Tensor, indexes: torch.Tensor, remove_clashed: bool = False, nested: bool = True):
        scores = scores.sigmoid()
        # 这里的scores也是没有特殊字符的
        # 按照论文代码里的解码方式是上下三角取平均
        # scores = (scores.transpose(1, 2) + scores)/2
        scores: np.ndarray = scores.detach().cpu().numpy()

        length: np.ndarray = indexes.detach().cpu().numpy()
        length = length.max(-1)

        span_mask = (scores.max(-1) > self.span_threshold)
        argmax = scores.argmax(-1)
        indexes = np.where(span_mask)
        entities = [[] for _ in range(scores.shape[0])]
        # 同labels一样没有特殊的标签
        # 将预测实体append到entities中
        for batch_idx, x, y in zip(*indexes):
            if x >= 0 and x < length[batch_idx] and y >= 0 and y < length[batch_idx] and x <= y:
                # (start, end, label_idx, score)
                entities[batch_idx].append(
                    (x, y, argmax[batch_idx, x, y] + 1, scores[batch_idx, x, y, argmax[batch_idx, x, y]]))
        # 对每一个batch, 按label_score的降序排列
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
        # 转换为set
        for batch_idx in range(len(entities)):
            entities[batch_idx] = set(
                map(lambda x: (x[0], x[1], x[2]), entities[batch_idx]))
        return entities

    def forward(self, input_ids: torch.Tensor, bpe_len: torch.Tensor, indexes: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        # input_ids 就是常规的input_ids, [batch_size, seq_length, hidden_dim]
        # bpe_len 是flat tokens和[CLS]和[SEP]的长度, 不包括[PAD] [batch_size]
        # indexes 是每个字的坐标[0,1,...], [batch_size, seq_length, hidden_dim]
        # matrix [batch_size, seq_length, seq_length, num_labels] 的0，1矩阵
        attention_mask = seq_len_to_mask(bpe_len)  # bsz x length x length
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_states = outputs['last_hidden_state']
        # 这里的效果其实跟W2NER是一样的，就是pieces2word
        # 所有index为0的标签会被选取包含最大的hidden_dim的token, 放置在第0位, 即[CLS], [SEP]和[PAD]的标签
        # 所有index相同的标签会被选取包含最大的hidden_dim的token, 放置在第index位
        # 其余位置补0
        # WARN: 这里会去除前后两个token，因此labels要提前去除前后两个token
        state = scatter_max(last_hidden_states, index=indexes, dim=1)[
            0][:, 1:]  # bsz x word_len x hidden_size
        # 真实的文本-标签对长度
        lengths, _ = indexes.max(dim=-1)

        # 1. state先传进head和tail的MLP压一下维度得到头尾特征
        head_state = self.head_mlp(state)
        tail_state = self.tail_mlp(state)

        # 2. 进单头还是多头
        if hasattr(self, 'U'):
            scores1 = torch.einsum(
                'bxi, oij, byj -> boxy', head_state, self.U, tail_state) # [batch_size, out_dim , word_len, word_len]
        else:
            # [batch_size, out_dim, word_len, word_len]
            scores1 = self.multi_head_biaffine(head_state, tail_state)
        
        # 3. head 和 tail 自我扩展成word_len*2后将hidden_state拼接并加入偏置项和相对距离positional embedding.
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
                               self.W)  # bsz x dim x L x L
        scores = scores2 + scores1   # bsz x dim x L x L

        if hasattr(self, 'cnn'):
            mask = seq_len_to_mask(lengths)  # bsz x length x length
            mask = mask[:, None] * mask.unsqueeze(-1)
            pad_mask = mask[:, None].eq(0)
            u_scores = scores.masked_fill(pad_mask, 0)
            if self.logit_drop != 0:
                u_scores = F.dropout(
                    u_scores, p=self.logit_drop, training=self.training)
            # bsz, num_label, max_len, max_len = u_scores.size()
            u_scores = self.cnn(u_scores, pad_mask)
            scores = u_scores + scores
        
        # 把dim作为尾部对准fc
        scores = self.down_fc(scores.permute(0, 2, 3, 1))

        assert scores.size(-1) == labels.size(-1)

        loss = None
        if labels is not None:
            flat_scores = scores.reshape(-1)
            flat_matrix = labels.reshape(-1)
            mask = flat_matrix.ne(-100).float().view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(
                flat_scores, flat_matrix.float(), reduction='none')
            loss = ((flat_loss.view(input_ids.size(0), -1)*mask).sum(dim=-1)).mean()

        return loss, scores


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
        :param mask: bsz x dim x max_len x max_len, 为1的地方为pad
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
        _x = x  # 用作residual
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
        # b: bsz, l: seq_len, h: head_num, x: in_head_dim, y: In_head_dim, d: out_head_dim, k: out_dim
        w = torch.einsum('blhx,hdxy,bkhy->bhdlk', h, self.W, v)
        # [batch_size, out_dim, seq_len, seq_len]
        w = w.reshape(bsz, self.out_dim, max_len, max_len)
        return w
