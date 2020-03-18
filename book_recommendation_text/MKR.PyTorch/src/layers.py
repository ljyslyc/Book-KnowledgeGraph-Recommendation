import torch
from torch import nn
from torch.nn.parameter import Parameter
from abc import abstractmethod

LAYER_IDS = {}

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, chnl=8):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = nn.ReLU()
        self.drop_layer = nn.Dropout(p=self.dropout) # Pytorch drop: ratio to zeroed
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, inputs):
        x = self.drop_layer(inputs)
        output = self.fc(x)
        return self.act(output)

class Bias(nn.Module):
    def __init__(self, dim):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        x = x + self.bias
        return x


class CrossCompressUnit(nn.Module):
    def __init__(self, dim):
        super(CrossCompressUnit, self).__init__()
        self.dim = dim
        self.fc_vv = nn.Linear(dim, 1, bias=False)
        self.fc_ev = nn.Linear(dim, 1, bias=False)
        self.fc_ve = nn.Linear(dim, 1, bias=False)
        self.fc_ee = nn.Linear(dim, 1, bias=False)

        self.bias_v = Bias(dim)
        self.bias_e = Bias(dim)

        # self.fc_v = nn.Linear(dim, dim)
        # self.fc_e = nn.Linear(dim, dim)

    def forward(self, inputs):
        v, e = inputs

        # [batch_size, dim, 1], [batch_size, 1, dim]
        v = torch.unsqueeze(v, 2)
        e = torch.unsqueeze(e, 1)

        # [batch_size, dim, dim]
        c_matrix = torch.matmul(v, e)
        c_matrix_transpose = c_matrix.permute(0,2,1)

        # [batch_size * dim, dim]
        c_matrix = c_matrix.view(-1, self.dim)
        c_matrix_transpose = c_matrix_transpose.contiguous().view(-1, self.dim)

        # [batch_size, dim]
        v_intermediate = self.fc_vv(c_matrix) + self.fc_ev(c_matrix_transpose)
        e_intermediate = self.fc_ve(c_matrix) + self.fc_ee(c_matrix_transpose)
        v_intermediate = v_intermediate.view(-1, self.dim)
        e_intermediate = e_intermediate.view(-1, self.dim)

        v_output = self.bias_v(v_intermediate)
        e_output = self.bias_e(e_intermediate)

        # v_output = self.fc_v(v_intermediate)
        # e_output = self.fc_e(e_intermediate)


        return v_output, e_output

