
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss


class BM3(GeneralRecommender):
    def __init__(self, config, dataset):
        super(BM3, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.dropout = config['dropout']

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()

        nn.init.xavier_normal_(self.predictor.weight)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.image_trs.weight)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_trs.weight)

    def get_norm_adj_mat(self, interaction_matrix):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def forward(self):
        h = self.item_id_embedding.weight

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def calculate_loss(self, interactions):
        # online network
        u_online_ori, i_online_ori = self.forward()
        t_feat_online, v_feat_online = None, None
        if self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)

        with torch.no_grad():  # 停止梯度更新，这样在下面的操作中不会计算梯度，节省内存和计算资源
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()  # 复制在线用户和物品的原始特征向量
            u_target.detach()  # 分离用户目标特征向量，使其不参与梯度计算
            i_target.detach()  # 分离物品目标特征向量，使其不参与梯度计算
            u_target = F.dropout(u_target, self.dropout)  # 对用户目标特征向量应用Dropout，以增加模型的鲁棒性
            i_target = F.dropout(i_target, self.dropout)  # 对物品目标特征向量应用Dropout，以增加模型的鲁棒性

            if self.t_feat is not None:  # 检查时间特征是否存在
                t_feat_target = t_feat_online.clone()  # 复制在线时间特征向量
                t_feat_target = F.dropout(t_feat_target, self.dropout)  # 对时间特征向量应用Dropout，以增加模型的鲁棒性

            if self.v_feat is not None:  # 检查视觉特征是否存在
                v_feat_target = v_feat_online.clone()  # 复制在线视觉特征向量
                v_feat_target = F.dropout(v_feat_target, self.dropout)  # 对视觉特征向量应用Dropout，以增加模型的鲁棒性

        # 预测用户和物品的在线特征向量
        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)

        # 获取交互数据中的用户和物品索引
        users, items = interactions[0], interactions[1]

        # 根据用户和物品索引提取相应的在线特征和目标特征
        u_online = u_online[users, :]  # 提取在线用户特征
        i_online = i_online[items, :]  # 提取在线物品特征
        u_target = u_target[users, :]  # 提取目标用户特征
        i_target = i_target[items, :]  # 提取目标物品特征

        # 初始化各类损失为0
        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0

        if self.t_feat is not None:  # 检查时间特征是否存在
            t_feat_online = self.predictor(t_feat_online)  # 通过预测器更新在线时间特征
            t_feat_online = t_feat_online[items, :]  # 提取更新后的在线时间特征
            t_feat_target = t_feat_target[items, :]  # 提取目标时间特征
            # 计算时间特征和物品目标特征的余弦相似度损失
            loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
            # 计算时间特征和目标时间特征的余弦相似度损失
            loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()

        if self.v_feat is not None:  # 检查视觉特征是否存在
            v_feat_online = self.predictor(v_feat_online)  # 通过预测器更新在线视觉特征
            v_feat_online = v_feat_online[items, :]  # 提取更新后的在线视觉特征
            v_feat_target = v_feat_target[items, :]  # 提取目标视觉特征
            # 计算视觉特征和物品目标特征的余弦相似度损失
            loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
            # 计算视觉特征和目标视觉特征的余弦相似度损失
            loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()

        # 计算用户在线特征和物品目标特征的余弦相似度损失
        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        # 计算物品在线特征和用户目标特征的余弦相似度损失
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        # 返回总损失，包括余弦相似度损失、正则化损失和对比损失
        return (loss_ui + loss_iu).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) + \
            self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt).mean()

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, i_online = self.forward()
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)
        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return score_mat_ui

