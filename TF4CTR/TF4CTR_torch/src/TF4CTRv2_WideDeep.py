# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch, os
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, LogisticRegression


# Full TF4CTR
class TF4CTRv2_WideDeep(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="TF4CTRv2_WideDeep",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 c=0.8,
                 gamma=2,
                 alpha=0.2,
                 SSM=None,
                 DFM=None,
                 easy_hidden_units=[64, 64, 64],
                 hard_hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(TF4CTRv2_WideDeep, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       embedding_regularizer=embedding_regularizer,
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.c = c
        self.gamma = gamma
        self.alpha = alpha
        input_dim = feature_map.sum_emb_out_dim()
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.easy_encoder = LogisticRegression(feature_map, use_bias=False)
        self.hard_encoder = MLP_Block(input_dim=input_dim,
                                      output_dim=None,
                                      hidden_units=hard_hidden_units,
                                      hidden_activations=hidden_activations,
                                      output_activation=None,
                                      dropout_rates=net_dropout,
                                      batch_norm=batch_norm)

        if DFM == 'WSF':
            self.DFM = WSF(sim_encoder_dim=1, com_encoder_dim=hard_hidden_units[-1])
        elif DFM == 'VF':
            self.DFM = VF(sim_encoder_dim=1, com_encoder_dim=hard_hidden_units[-1])
        elif DFM == 'CF':
            self.DFM = CF(sim_encoder_dim=1, com_encoder_dim=hard_hidden_units[-1])
        elif DFM == 'MoEF':
            self.DFM = MoEF(sim_encoder_dim=1, com_encoder_dim=hard_hidden_units[-1])
        else:
            self.DFM = sum_fusion(sim_encoder_dim=1, com_encoder_dim=hard_hidden_units[-1])

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        sim_encoder_out = self.easy_encoder(X)
        hard_out = self.embedding_layer(X, dynamic_emb_dim=True)
        com_encoder_out = self.hard_encoder(hard_out)
        y_pred, y_sim, y_com = self.DFM(sim_encoder_out=sim_encoder_out, com_encoder_out=com_encoder_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred, 'y_sim': y_sim, 'y_com': y_com}
        return return_dict

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        y_easy = self.output_activation(return_dict["y_sim"])
        y_hard = self.output_activation(return_dict["y_com"])
        TFLoss = self.TFLoss(y_easy=y_easy, y_hard=y_hard,
                             y_true=y_true, c=self.c,
                             gamma=self.gamma, alpha=self.alpha, reduction='mean')
        loss = loss + TFLoss
        return loss

    # Since the AUC is not sensitive to positive and negative samples,
    # we tried to remove alpha, which improved the effect
    def TFLoss(self, y_easy, y_hard, y_true, c=0.8, gamma=2, alpha=0.25, reduction='mean'):
        assert type is not None, "Missing type parameter. You can choose between easy or hard."
        # y_pred should be 0~1 value
        # EASY LOSS
        Logloss = self.loss_fn(y_easy, y_true, reduction='none')
        p_t = y_true * y_easy + (1 - y_true) * (1 - y_easy)
        modulating_factor = (c + p_t) ** gamma
        easy_loss = Logloss * modulating_factor
        # HARD LOSS
        Logloss = self.loss_fn(y_hard, y_true, reduction='none')
        p_t = y_true * y_hard + (1 - y_true) * (1 - y_hard)
        modulating_factor = ((2 - c) - p_t) ** gamma
        hard_loss = Logloss * modulating_factor

        if reduction == 'mean':
            easy_loss = easy_loss.mean()
            hard_loss = hard_loss.mean()
        elif reduction == 'sum':
            easy_loss = easy_loss.sum()
            hard_loss = hard_loss.sum()

        return alpha * easy_loss + (1 - alpha) * hard_loss

class WSF(nn.Module):
    def __init__(self, sim_encoder_dim, com_encoder_dim):
        super(WSF, self).__init__()
        # self.fc1 = nn.Linear(sim_encoder_dim, 1)
        self.fc2 = nn.Linear(com_encoder_dim, 1)
        self.w = nn.Parameter(torch.empty(2, 1).fill_(0.5), requires_grad=True)

    def forward(self, sim_encoder_out, com_encoder_out):
        y_pred_1 = sim_encoder_out
        y_pred_2 = self.fc2(com_encoder_out)
        y_pred = torch.matmul(torch.cat([y_pred_1, y_pred_2], dim=-1), self.w)
        return y_pred, y_pred_1, y_pred_2


class VF(nn.Module):
    def __init__(self, sim_encoder_dim, com_encoder_dim):
        super(VF, self).__init__()
        self.fc1 = nn.Linear(sim_encoder_dim, 1)
        self.fc2 = nn.Linear(com_encoder_dim, 1)
        self.pi1 = nn.Linear(sim_encoder_dim, 1)
        self.pi2 = nn.Linear(sim_encoder_dim, 1)

    def forward(self, sim_encoder_out, com_encoder_out):
        y_pred_1 = self.fc1(sim_encoder_out)
        y_pred_2 = self.fc2(com_encoder_out)
        pi1 = self.pi1(sim_encoder_out)
        pi2 = self.pi2(com_encoder_out)
        pi = torch.cat([pi1, pi2], dim=-1)
        y_pred = torch.cat([y_pred_1, y_pred_2], dim=-1)
        p = F.gumbel_softmax(pi, tau=0.8)
        y_pred = torch.sum(p * y_pred, dim=-1, keepdim=True)
        return y_pred, y_pred_1, y_pred_2


class CF(nn.Module):
    def __init__(self, sim_encoder_dim, com_encoder_dim):
        super(CF, self).__init__()
        self.fc = nn.Linear(sim_encoder_dim + com_encoder_dim, 1)
        self.fc1 = nn.Linear(sim_encoder_dim, 1)
        self.fc2 = nn.Linear(com_encoder_dim, 1)

    def forward(self, sim_encoder_out, com_encoder_out):
        total_encoder_out = torch.cat([sim_encoder_out, com_encoder_out], dim=-1)
        y_pred = self.fc(total_encoder_out)
        y_pred_1 = self.fc1(sim_encoder_out)
        y_pred_2 = self.fc2(com_encoder_out)
        return y_pred, y_pred_1, y_pred_2


class MoEF(nn.Module):
    def __init__(self, sim_encoder_dim, com_encoder_dim):
        super(MoEF, self).__init__()
        input_dim = sim_encoder_dim + com_encoder_dim
        self.gate = nn.Sequential(nn.Linear(input_dim, 2),
                                  nn.Softmax())
        self.MOE = nn.Linear(input_dim, 2)
        self.fc1 = nn.Linear(sim_encoder_dim, 1)
        self.fc2 = nn.Linear(com_encoder_dim, 1)

    def forward(self, sim_encoder_out, com_encoder_out):
        total_encoder_out = torch.cat([sim_encoder_out, com_encoder_out], dim=-1)
        g = self.gate(total_encoder_out)
        m = self.MOE(total_encoder_out)
        y_pred = torch.sum(g * m, dim=-1, keepdim=True)
        y_pred_1 = self.fc1(sim_encoder_out)
        y_pred_2 = self.fc2(com_encoder_out)
        return y_pred, y_pred_1, y_pred_2


class sum_fusion(nn.Module):
    def __init__(self, sim_encoder_dim, com_encoder_dim):
        super(sum_fusion, self).__init__()
        self.fc1 = nn.Linear(sim_encoder_dim, 1)
        self.fc2 = nn.Linear(com_encoder_dim, 1)

    def forward(self, sim_encoder_out, com_encoder_out):
        y_pred_1 = self.fc1(sim_encoder_out)
        y_pred_2 = self.fc2(com_encoder_out)
        return y_pred_1 + y_pred_2, y_pred_1, y_pred_2


class MMoE_Layer(nn.Module):
    def __init__(self, input_dim, expert_hidden_units, gate_hidden_units, net_dropout, batch_norm,
                 num_experts=2, num_tasks=2, hidden_activations="ReLU"):
        super(MMoE_Layer, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                                hidden_units=expert_hidden_units,
                                                hidden_activations=hidden_activations,
                                                output_activation=None,
                                                dropout_rates=net_dropout,
                                                batch_norm=batch_norm) for _ in range(self.num_experts)])
        self.gate = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                             hidden_units=gate_hidden_units,
                                             output_dim=num_experts,
                                             hidden_activations=hidden_activations,
                                             output_activation=None,
                                             dropout_rates=net_dropout,
                                             batch_norm=batch_norm) for _ in range(self.num_tasks)])
        self.gate_activation = nn.Softmax()

    def forward(self, x):
        experts_output = torch.stack([self.experts[i](x) for i in range(self.num_experts)],
                                     dim=1)  # (?, num_experts, dim)
        mmoe_output = []
        for i in range(self.num_tasks):
            gate_output = self.gate[i](x)
            if self.gate_activation is not None:
                gate_output = self.gate_activation(gate_output)  # (?, num_experts)
            mmoe_output.append(torch.sum(torch.multiply(gate_output.unsqueeze(-1), experts_output), dim=1))
        return mmoe_output
