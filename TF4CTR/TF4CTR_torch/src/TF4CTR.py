# =========================================================================
# Copyright (C) 2024 salmon1802li@gmail.com
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
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block

# Only TF Loss has been added
class TF4CTR(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="TF4CTR",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 c=0.8,
                 gamma=2,
                 alpha=0.2,
                 easy_hidden_units=[64, 64, 64],
                 hard_hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(TF4CTR, self).__init__(feature_map,
                                     model_id=model_id,
                                     gpu=gpu,
                                     embedding_regularizer=embedding_regularizer,
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.c = c
        self.gamma = gamma
        self.alpha = alpha
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()
        self.easy_net = MLP_Block(input_dim=input_dim,
                                  output_dim=1,
                                  hidden_units=easy_hidden_units,
                                  hidden_activations=hidden_activations,
                                  output_activation=None,
                                  dropout_rates=net_dropout,
                                  batch_norm=batch_norm)
        self.hard_net = MLP_Block(input_dim=input_dim,
                                  output_dim=1,
                                  hidden_units=hard_hidden_units,
                                  hidden_activations=hidden_activations,
                                  output_activation=None,
                                  dropout_rates=net_dropout,
                                  batch_norm=batch_norm)
        self.fc = nn.Parameter(torch.empty(2, 1).fill_(0.5), requires_grad=True)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_embeddings = self.embedding_layer(X, dynamic_emb_dim=True)
        easy_net_out = self.easy_net(feature_embeddings)
        hard_net_out = self.hard_net(feature_embeddings)
        y_pred = torch.matmul(torch.cat([easy_net_out, hard_net_out], dim=-1), self.fc)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred, 'y_easy': easy_net_out, 'y_hard': hard_net_out}
        return return_dict

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        y_easy = self.output_activation(return_dict["y_easy"])
        y_hard = self.output_activation(return_dict["y_hard"])
        TFLoss = self.TFLoss(y_easy=y_easy, y_hard=y_hard,
                             y_true=y_true, c=self.c,
                             gamma=self.gamma, alpha=self.alpha, reduction='mean')
        loss = loss + TFLoss
        return loss

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

    # # Since the AUC is not sensitive to positive and negative samples,
    # # we tried to remove alpha, which improved the effect
    # def TFLoss(self, y_easy, y_hard, y_true, c=0.8, gamma=2, alpha=0.25, reduction='mean'):
    #     assert type is not None, "Missing type parameter. You can choose between easy or hard."
    #     # y_pred should be 0~1 value
    #     # EASY LOSS
    #     Logloss = self.loss_fn(y_easy, y_true, reduction='none')
    #     p_t = (y_true * y_easy + (1 - y_true) * (1 - y_easy)).detach()
    #     modulating_factor = (c + p_t) ** gamma
    #     easy_loss = Logloss * modulating_factor
    #     # HARD LOSS
    #     Logloss = self.loss_fn(y_hard, y_true, reduction='none')
    #     p_t = (y_true * y_hard + (1 - y_true) * (1 - y_hard)).detach()
    #     modulating_factor = ((2 - c) - p_t) ** gamma
    #     hard_loss = Logloss * modulating_factor
    #
    #     if reduction == 'mean':
    #         easy_loss =  easy_loss.mean()
    #         hard_loss = hard_loss.mean()
    #     elif reduction == 'sum':
    #         easy_loss = easy_loss.sum()
    #         hard_loss = hard_loss.sum()
    #
    #     return alpha * easy_loss + (1 - alpha) * hard_loss
