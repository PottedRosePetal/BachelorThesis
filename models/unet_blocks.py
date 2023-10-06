import torch
import torch.nn as nn
from torch.nn import Module, Linear
import time
   
class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x:torch.Tensor):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret
    
# class ConcatSquashLinearPCA(Module):
#     def __init__(self, dim_in, dim_out, dim_ctx):
#         super(ConcatSquashLinearPCA, self).__init__()
#         self._layer = Linear(dim_in, dim_out)
#         self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
#         self._hyper_gate = Linear(dim_ctx, dim_out)

#     def forward(self, ctx, x:torch.Tensor):
#         U, S, V = torch.pca_lowrank(x)  
#         # V columns represent the principal directions
#         # S∗∗2/(m−1) contains the eigenvalues of ATA/(m−1)ATA/(m−1) which is the covariance of A when center=True is provided.
#         # U is m x q matrix
#         # Documentation: https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html
        
#         gate = torch.sigmoid(self._hyper_gate(ctx))
#         bias = self._hyper_bias(ctx)
#         # if x.dim() == 3:
#         #     gate = gate.unsqueeze(1)
#         #     bias = bias.unsqueeze(1)
#         ret = self._layer(x) * gate + bias
#         return ret
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        # Define the main path with two 1D convolutional layers
        self.main_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_channels),
            torch.nn.functional.leaky_relu(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_channels)
        )

        # Define the shortcut connection
        self.shortcut = nn.Sequential()

    def forward(self, x):
        # Pass the input through the main path
        out = self.main_path(x)

        # Pass the input through the shortcut connection
        shortcut = self.shortcut(x)

        # Add the main path output and the shortcut output
        out += shortcut

        return out


class ConcatSquashLinearPCA(Module):
    def __init__(self, dim_in, dim_out, dim_ctx, pca_rank = 64):
        super(ConcatSquashLinearPCA, self).__init__()
        self._pca_linear = nn.Linear(dim_in, pca_rank)  # Linear layer for principal directions
        self._layer = nn.Linear(dim_in + pca_rank, dim_out)  # Combined input size
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x: torch.Tensor):
        strt_time = time.time()
        U, S, V = torch.pca_lowrank(x)
        # Documentation: https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html
        
        pca_projection = self._pca_linear(U)
        combined_input = torch.cat((x, pca_projection), dim=-1)
        print(time.time() - strt_time)
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        
        ret = self._layer(combined_input) * gate + bias
        return ret

class ConcatSquashDoubleLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx_1, dim_ctx_2):
        super(ConcatSquashDoubleLinear, self).__init__()

        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias1 = Linear(dim_ctx_1, dim_out, bias=False)
        self._hyper_bias2 = Linear(dim_ctx_2, dim_out, bias=False)
        self._hyper_gate1 = Linear(dim_ctx_1, dim_out)
        self._hyper_gate2 = Linear(dim_ctx_2, dim_out)


    def forward(self, ctx1, ctx2, x):
        gate1 = torch.sigmoid(self._hyper_gate1(ctx1))
        gate2 = torch.sigmoid(self._hyper_gate2(ctx2))
        bias1 = self._hyper_bias1(ctx1)
        bias2 = self._hyper_bias2(ctx2)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret1 = self._layer(x) * gate1 + bias1
        ret2 = self._layer(x) * gate2 + bias2
        return (ret1 + ret2)/2
