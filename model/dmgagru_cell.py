'''
Date: 2021-01-13 20:00:19
LastEditTime: 2021-01-13 20:59:46
Description: The module of DMGA_GRU
FilePath: /DMGAN/model/dmgagru_cell.py
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import utils

        
class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str, device: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type
        self.device = device

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=self.device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=self.device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class DMGAGRUcell(torch.nn.Module):
    def __init__(self, device, num_units, adj_mx, max_diffusion_step, num_nodes, alpha, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True, k_hop=3):
        """

        :param num_units: ->  rnn_units
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self.device = device
        self._use_gc_for_ru = use_gc_for_ru
        supports = []
        """ para in multi-hop """
        self.alpha = alpha
        self.k_hop = k_hop

        if filter_type == "laplacian":
            #print(filter_type)
            supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
            supports += utils.calculate_topalogy_laplacian(adj_mx, k=1)

        else:
            supports.append(utils.calculate_scaled_laplacian(adj_mx))

        for support in supports:
            self._supports.append(self._build_sparse_matrix(support, self.device))

        self._gconv_params = LayerParams(self, 'gconv', self.device)

    @staticmethod
    def _build_sparse_matrix(L, device):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L

    def forward(self, inputs, hx, time_axis, adp):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, hx, output_size, time_axis=time_axis, bias_start=1.0, adp=adp))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        if self._use_gc_for_ru:
            c = self._gconv(inputs, r * hx, self._num_units, time_axis=time_axis, adp=adp)
        else:
            c = self._fc(inputs, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)
    
    @staticmethod
    def get_multi_hop_adj(adj, k_hop, alpha):
        if k_hop == 1:
            return [adj]
        elif k_hop == 2:
            return [(1 - alpha) * alpha * adj, ((1 - alpha) ** 2) * (torch.bmm(adj, adj))]
        elif k_hop == 3:
            return [(1 - alpha) * alpha * adj, ((1 - alpha) ** 2) * alpha * (torch.bmm(adj, adj)), 
                     ((1 - alpha) ** 3) * (torch.bmm(torch.bmm(adj, adj), adj))]
        elif k_hop == 4:
            return [(1 - alpha) * alpha * adj, ((1 - alpha) ** 2) * alpha * (torch.bmm(adj, adj)), 
                      ((1 - alpha) ** 3) * alpha * (torch.bmm(torch.bmm(adj, adj), adj)), ((1 - alpha) ** 4) * (torch.bmm(torch.bmm(torch.bmm(adj, adj), adj), adj))]


    def _gconv(self, inputs, state, output_size, time_axis=0, bias_start=0.0, adp=None):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1
  
        adp_lst = self.get_multi_hop_adj(adp, self.k_hop, self.alpha)

        for adp_ in adp_lst:
            x_temp = torch.reshape(x0, shape=[self._num_nodes, input_size, batch_size]).permute(2, 0, 1)
            x1 = torch.bmm(adp_, x_temp).permute(1, 2, 0).reshape(self._num_nodes, input_size * batch_size)
            x = self._concat(x, x1)
        

        num_matrices = (len(self._supports) + len(adp_lst)) * self._max_diffusion_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
