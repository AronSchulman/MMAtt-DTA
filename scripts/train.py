import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize, scale, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from typing import Optional, Any, Union, Callable
from lifelines.utils import concordance_index
import wandb
import random
import pickle
import math
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

import ray
from ray import air, train, tune
from ray.air import session, ScalingConfig, RunConfig, CheckpointConfig
import ray.train.torch
from ray.train.torch import TorchTrainer, TorchConfig, TorchCheckpoint
from ray.air.checkpoint import Checkpoint
from ray.air.integrations.wandb import setup_wandb, WandbLoggerCallback
from ray.tune import Trainable, CLIReporter
from ray.tune.logger import DEFAULT_LOGGERS
from pprint import pprint

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

### periodic embeddings for improved results in tabular deep learning ###

class PeriodicEmbeddings(nn.Module):
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
        cat_embed_tensor_group = torch.empty((x_cat.shape[0], num_cats, self.cfg["embed_dim"])).to(torch.device(f"cuda:{session.get_local_rank()}"))
        for i in range(num_cats):
            cat_embed_tensor_group[:,i,:] = self.cat_embeds[i](x_cat[:,i])
        x_cont = self.cont_embed(x_cont)
        x_both = torch.cat((cat_embed_tensor_group, x_cont), dim=1) if num_cats > 0 else x_cont
        x_cls = self.cls_embed(x_both)
        x_transformed = self.tr_encoder(self.custom_encoder_layer(x_cls))
        
        return x_transformed
    
class ExperimentalDTINN(nn.Module):
    def __init__(self, cfg, model_comp, model_prot):
        super(ExperimentalDTINN, self).__init__()
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

class DrugTargetDataset(Dataset):
    def __init__(self, comp_x_cat, comp_x_cont, prot_x_cat, prot_x_cont, labels):
        self.comp_x_cat = torch.tensor(comp_x_cat.values, dtype=torch.int64)
        self.prot_x_cat = torch.tensor(prot_x_cat.values, dtype=torch.int64)
        
        self.comp_x_cont = torch.tensor(comp_x_cont.values, dtype=torch.float32)
        self.prot_x_cont = torch.tensor(prot_x_cont.values, dtype=torch.float32)
        
        self.labels = torch.tensor(labels.values, dtype=torch.float32).reshape(-1,1)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        comp_cat_feats = self.comp_x_cat[idx]
        comp_cont_feats = self.comp_x_cont[idx]
        prot_cat_feats = self.prot_x_cat[idx]
        prot_cont_feats = self.prot_x_cont[idx]
        label = self.labels[idx]

        return comp_cat_feats, comp_cont_feats, prot_cat_feats, prot_cont_feats, label
        
    
def plot_correlation(ys, preds):
    sns.set_style("ticks")
    fig, ax = plt.subplots(1, 2, sharey=True, figsize = (10,5), dpi=100)
    
    for i in range(2):
        if i == 0:
            ax[i].scatter(ys, preds, s=3, alpha=0.5, color="firebrick")
        else:
            sns.kdeplot(x=ys.flatten(), y=preds.flatten(), cmap=sns.light_palette('firebrick', as_cmap=True), fill=True, ax=ax[1])
        straight = np.linspace(5, 10, num=100)
        ax[i].plot(straight, straight, color="gray", alpha=0.5, linewidth=2.4)
    
        ax[i].tick_params(axis='both', which='major', labelsize=15)
        ax[i].tick_params(axis='both', which='minor', labelsize=15)
        ax[i].xaxis.set_ticks(np.arange(5, 10.1, 0.5))
        ax[i].yaxis.set_ticks(np.arange(5, 10.1, 0.5))
    
    fig.supxlabel("True interaction strength", fontsize=18)
    fig.supylabel("Predicted interaction strength", fontsize=18)
    
    
    fig.patch.set_facecolor('white')
    sns.despine()
    plt.tight_layout()

    wandb.log({"Test scatter plot":wandb.Image(fig)})
    

def test_loop(dataloader, model, loss_fn, epoch):
    local_test_rank = session.get_local_rank()
    world_size = session.get_world_size()
    test_device = torch.device(f"cuda:{local_test_rank}")
    model.eval()
    preds = torch.empty(0, dtype=torch.int64, device=test_device)
    ys = torch.empty(0, dtype=torch.int64, device=test_device)
    with torch.no_grad():
        for (comp_x_cat, comp_x_cont, prot_x_cat, prot_x_cont, y) in dataloader:
            pred = model(comp_x_cat, comp_x_cont, prot_x_cat, prot_x_cont)
            ys = torch.concat([ys, y])
            preds = torch.concat([preds, pred])
      
    ys = ys.flatten()     
    preds = preds.flatten()
    ys_all_out = torch.zeros(len(ys) * world_size, dtype=torch.float32, device=test_device)
    preds_all_out = torch.zeros(len(preds) * world_size, dtype=torch.float32, device=test_device)
    dist.all_gather_into_tensor(ys_all_out, ys)
    dist.all_gather_into_tensor(preds_all_out, preds)
    
    test_loss = loss_fn(preds_all_out, ys_all_out)
    
    test_loss = float(test_loss.detach().cpu().numpy())
    ys_all_out = ys_all_out.detach().cpu().numpy()
    preds_all_out = preds_all_out.detach().cpu().numpy()
    spear = spearmanr(preds_all_out, ys_all_out).correlation
    rmse = mean_squared_error(ys_all_out, preds_all_out, squared=False)
    cindex = concordance_index(ys_all_out, preds_all_out)
    
    # if local_test_rank == 0 and epoch % 10 == 0:
        # plot_correlation(ys_all_out, preds_all_out)
        

    return test_loss, spear, rmse, cindex
    
def create_cat_cont_split(df):
    cats = []
    conts = []
    for col in df.columns:
        if "categorical" in col:
            cats.append(col)
        elif "continuous" in col:
            conts.append(col)
        else:
            print(f"Column {col} was not assigned to cat or cont.")
    df_cat = df[cats]
    df_cont = df[conts]
    
    return df_cat, df_cont
    
def prepare_dataloader(df_x, df_y, config):
    cols = df_x.columns
    col_prots = []
    col_comps = []
    col_other = []
    for c in cols:
        if "protein" in c:
            col_prots.append(c)
        elif "compound":
            col_comps.append(c)
        else:
            col_other.append(c)
            print(f"Column {col} was not assigned to protein or compound.")
    df_comp_x = df_x[col_comps]
    df_prot_x = df_x[col_prots]
    
    df_comp_x_cat, df_comp_x_cont = create_cat_cont_split(df_comp_x)
    df_prot_x_cat, df_prot_x_cont = create_cat_cont_split(df_prot_x)

    data_set = DrugTargetDataset(df_comp_x_cat, df_comp_x_cont, df_prot_x_cat, df_prot_x_cont, df_y)
    data_loader = DataLoader(data_set, batch_size=config["batch_size"])
    data_loader = train.torch.prepare_data_loader(data_loader)
    
    shapes = [df_comp_x_cont.shape[1], df_comp_x_cat.shape[1], df_prot_x_cont.shape[1], df_prot_x_cat.shape[1]] # return for building HalfBlocks
    
    return data_loader, shapes
    

    
def run_train(config):
    wandb = setup_wandb(config, rank_zero_only=True)
    
    df_train = pd.read_csv(config["train_file"])
    df_validate = pd.read_csv(config["val_file"])
    df_train_x, df_train_y = df_train.drop(["interaction_strength", "pchembl_value"], axis=1), df_train.pchembl_value
    df_validate_x, df_validate_y = df_validate.drop(["interaction_strength", "pchembl_value"], axis=1), df_validate.pchembl_value
    

    # checkpoint = session.get_checkpoint()
    # start = 1 if not checkpoint else checkpoint.to_dict()["epoch"] + 1
    
    train_dataloader, shapes = prepare_dataloader(df_train_x, df_train_y, config)
    val_dataloader, _ = prepare_dataloader(df_validate_x, df_validate_y, config)
    
    loss_fn = LogCoshLoss()
    model_comp = HalfBlock(shapes[0], shapes[1], config, "compound")
    model_prot = HalfBlock(shapes[2], shapes[3], config, "protein")
    model = ExperimentalDTINN(config, model_comp, model_prot)
    model = train.torch.prepare_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    number_of_datapoints = len(df_train_x) / 4 # divide by num of gpus
    patience = 0
    best_test_loss = 1000
    for t in range(1, config["epochs"]):
        print(f"Epoch {t}")
        model.train()
        loss_total = 0
        for batch, (comp_x_cat, comp_x_cont, prot_x_cat, prot_x_cont, y) in enumerate(train_dataloader, 1):
            optimizer.zero_grad(set_to_none=True)
            pred = model(comp_x_cat, comp_x_cont, prot_x_cat, prot_x_cont)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            loss_total += loss * len(comp_x_cat)
     
        mean_train_loss = loss_total / number_of_datapoints
        test_loss, spear, rmse, cindex = test_loop(val_dataloader, model, loss_fn, t)
        
        train_loss_to_session = float(mean_train_loss.detach().cpu().numpy())
        
        session.report({"epoch" : t, "mean_train_loss" : train_loss_to_session, "test_loss" : test_loss, "test_correlation" : spear, "test_rmse" : rmse}, checkpoint=TorchCheckpoint.from_dict({"epoch" : t, "test_loss" : test_loss, "model_weights" : model.state_dict()}))
        wandb.log({f"{config['protein']} train cosh loss" : mean_train_loss, f"{config['protein']} test cosh loss" : test_loss, "Test Spearman" : spear, "Test RMSE" : rmse, "Test CI" : cindex})
        
        print("Test loss:", test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience = 0
        else:
            patience += 1
        
        if patience > config["max_patience"]:
            print(f"Patience ended at epoch {t}. No improvement for test loss after {config['max_patience']} epochs. Exiting.")
            break
        
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--protein_class", action="store", help="Provide a protein class for the model to be trained on.")
    parser.add_argument("-t", "--train_file", action="store", help="Provide train data file path.")
    parser.add_argument("-v", "--val_file", action="store", help="Provide validation data file path.")
    args = parser.parse_args()
    
    wandb.login()
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=40, num_gpus=4)
    
    with open('json_files/model_config_params.json', 'r') as json_file: # config for model architecture & hyperparams
        cfg_all = json.load(json_file)
    
    cfg = cfg_all[args.protein_class]
    cfg["train_file"] = args.train_file
    cfg["val_file"] = args.val_file
    
    ray_trainer = TorchTrainer(run_train,
                                train_loop_config=cfg,
                                run_config=RunConfig(name=f"{args.protein_class} random split", local_dir="train_results/", checkpoint_config=CheckpointConfig(num_to_keep=cfg["n_ensemble"], checkpoint_score_attribute="test_loss", checkpoint_score_order="min")),
                                torch_config=TorchConfig(backend='nccl'),
                                scaling_config=ScalingConfig(use_gpu=True, num_workers=4, resources_per_worker={"GPU" : 1, "CPU" : 9}))
                                
    
    
    results = ray_trainer.fit()
    pprint(results.metrics)
    
    wandb.finish()
    
if __name__ == "__main__":
    main()
    
    