import pandas as pd
import numpy as np
from typing import Optional, Any, Union, Callable
import random
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torch.nn.parameter import Parameter

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

class PeriodicEmbeddings(nn.Module):
    ### periodic embeddings for improved results in tabular deep learning ###
    # Source: https://github.com/Yura52/tabular-dl-num-embeddings/blob/e49e95c52f829ad0ab7d653e0776c2a84c03e261/lib/deep.py#L28
    def __init__(self, n_features: int, d_embedding: int, sigma: float) -> None:
        if d_embedding % 2:
            raise ValueError('d_embedding must be even')

        super().__init__()
        self.sigma = sigma
        self.coefficients = Parameter(Tensor(n_features, d_embedding // 2))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.coefficients, 0.0, self.sigma)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError('The input must have two dimensions')
        x = 2 * math.pi * self.coefficients[None] * x[..., None]
        return torch.cat([torch.cos(x), torch.sin(x)], -1)


class NLinear(nn.Module):
    def __init__(self, n_tokens: int, d_in: int, d_out: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = Parameter(Tensor(n_tokens, d_in, d_out))
        self.bias = Parameter(Tensor(n_tokens, d_out)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # This initialization is equivalent to that of torch.nn.Linear
        d_in = self.weight.shape[1]
        bound = 1 / math.sqrt(d_in)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                'The input must have three dimensions (batch_size, n_tokens, d_embedding)'
            )
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x
    
def _initialize_embeddings(weight: Tensor, d: int) -> None:
    d_sqrt_inv = 1 / math.sqrt(d)
    nn.init.uniform_(weight, a=-d_sqrt_inv, b=d_sqrt_inv)
    

class CLSEmbedding(nn.Module):
    def __init__(self, d_embedding: int) -> None:
        """
        Args:
            d_embedding: the size of the embedding
        """
        super().__init__()
        self.weight = Parameter(Tensor(d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _initialize_embeddings(self.weight, self.weight.shape[-1])

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError('The input must have three dimensions')
        if x.shape[-1] != len(self.weight):
            raise ValueError(
                'The last dimension of x must be equal to the embedding size'
            )
        return torch.cat([self.weight.expand(len(x), 1, -1), x], dim=1)

def make_plr_embeddings(
    n_features: int, d_embedding: int, d_periodic_embedding: int, sigma: float
) -> nn.Module:
    return nn.Sequential(
        PeriodicEmbeddings(n_features, d_periodic_embedding, sigma),
        NLinear(n_features, d_periodic_embedding, d_embedding),
        nn.ReLU(),
    )

class CustomTransformerEncoderLayer(nn.Module):
  
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False) -> torch.Tensor:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )


        x = src
        x = x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
        x = x + self._ff_block(x)

        return x


    # self-attention block
    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor], is_causal: bool = False) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)
    
class HalfBlock(nn.Module):
    def __init__(self, n_cont_features, n_cat_features, cfg, mode):
        super(HalfBlock, self).__init__()
        self.cfg = cfg
        tr_encoder_layer = nn.TransformerEncoderLayer(d_model=cfg["embed_dim"], nhead=8, dropout=cfg["att_dropout_half"], dim_feedforward=cfg["enc_FF_dim_half"], norm_first=True, batch_first=True)
        self.tr_encoder = nn.TransformerEncoder(tr_encoder_layer, num_layers=cfg["att_layers_half"])
        self.apply(self.init_weights)
        
        sigma = cfg["embed_sigma_" + mode]
        periodic_embed_dim = cfg["embed_dim_periodic_" + mode]
        self.cat_embeds = nn.ModuleList()
        for i in range(n_cat_features):
            self.cat_embeds.append(nn.Embedding(2, cfg["embed_dim"]))
        self.cont_embed = make_plr_embeddings(n_cont_features, cfg["embed_dim"], periodic_embed_dim, sigma)
        self.cls_embed = CLSEmbedding(cfg["embed_dim"])
        self.custom_encoder_layer = CustomTransformerEncoderLayer(d_model=cfg["embed_dim"], nhead=8, dropout=cfg["att_dropout_half"], dim_feedforward=cfg["enc_FF_dim_half"], batch_first=True) # only difference is no normalization as suggested in the paper https://proceedings.neurips.cc/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf
        

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)
            
    def forward(self, x_cat, x_cont):
        num_cats = x_cat.shape[1]
        cat_embed_tensor_group = torch.empty((x_cat.shape[0], num_cats, self.cfg["embed_dim"])).to(device)
        for i in range(num_cats):
            cat_embed_tensor_group[:,i,:] = self.cat_embeds[i](x_cat[:,i])
        x_cont = self.cont_embed(x_cont)
        x_both = torch.cat((cat_embed_tensor_group, x_cont), dim=1) if num_cats > 0 else x_cont
        x_cls = self.cls_embed(x_both)
        x_transformed = self.tr_encoder(self.custom_encoder_layer(x_cls))
        
        return x_transformed
    
class AttentionDTINN(nn.Module):
    def __init__(self, cfg, model_comp, model_prot):
        super(AttentionDTINN, self).__init__()
        tr_encoder_layer = nn.TransformerEncoderLayer(d_model=cfg["embed_dim"], nhead=8, dropout=cfg["att_dropout_full"], dim_feedforward=cfg["enc_FF_dim_full"], norm_first=True, batch_first=True)
        self.tr_encoder = nn.TransformerEncoder(tr_encoder_layer, num_layers=cfg["att_layers_full"])
        self.prediction = nn.Sequential(nn.LayerNorm(cfg["embed_dim"]), nn.ReLU(), nn.Linear(cfg["embed_dim"], 1))
        self.apply(self.init_weights)
        
        self.model_comp = model_comp
        self.model_prot = model_prot

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)
        
    def forward(self, comp_x_cat, comp_x_cont, prot_x_cat, prot_x_cont):
        x_comp = self.model_comp(comp_x_cat, comp_x_cont)
        x_prot = self.model_prot(prot_x_cat, prot_x_cont)
        x_transformed = self.tr_encoder(torch.cat((x_comp, x_prot), dim=1))
        x_cls_row = x_transformed[:,0,:]
        out = self.prediction(x_cls_row)
        
        return out
        