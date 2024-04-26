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

import logging
import numpy as np
import torch, os
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block

# Only TF Loss has been added
class TF4CTR_emb_grad(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="TF4CTR_emb_grad",
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
        super(TF4CTR_emb_grad, self).__init__(feature_map,
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
        self.grad = []
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
        # y_pred = easy_net_out + hard_net_out
        return_dict = {"y_pred": y_pred, 'y_sim': easy_net_out, 'y_com': hard_net_out,
                       "feature_embeddings": feature_embeddings}
        return return_dict

    def has_gradients(model):
        for param in model.parameters():
            if param.grad is not None:
                return True
        return False

    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs) -> object:
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self.train_epoch(data_generator)
            if epoch == 0:
                torch.save(self.grad, os.path.abspath(
                    os.path.join(self.model_dir, self.model_id + ".tf_emb_grad_norm_{}_epoch".format(self._epoch_index))))
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        loss, return_dict = self.get_total_loss(batch_data)
        # Enable retaining gradients
        feature_embeddings = return_dict['feature_embeddings'].retain_grad()
        loss.backward()
        if self.has_gradients():
            # Calculate gradient norms
            grad_norm_emb = return_dict["feature_embeddings"].grad.norm()
            self.grad.append(grad_norm_emb)
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss

    def get_total_loss(self, inputs):
        loss, return_dict = self.add_loss(inputs)
        total_loss = loss + self.add_regularization()
        return total_loss, return_dict

    # def add_loss(self, inputs):
    #     return_dict = self.forward(inputs)
    #     y_true = self.get_labels(inputs)
    #     y_pred = self.output_activation(return_dict["y_pred"])
    #     loss = self.loss_fn(y_pred, y_true, reduction='mean')
    #     loss = loss
    #     return loss, return_dict

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        y_pred = self.output_activation(return_dict["y_pred"])
        y_easy = self.output_activation(return_dict["y_sim"])
        y_hard = self.output_activation(return_dict["y_com"])
        loss = self.loss_fn(y_pred, y_true, reduction='mean')
        TFLoss = self.TFLoss(y_easy=y_easy, y_hard=y_hard,
                             y_true=y_true, c=self.c,
                             gamma=self.gamma, alpha=self.alpha, reduction='mean')
        loss = loss + TFLoss
        return loss, return_dict

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
