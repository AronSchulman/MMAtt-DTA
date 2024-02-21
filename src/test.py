import argparse
import pandas as pd
import numpy as np
import torch
import json
import pickle

from lifelines.utils import concordance_index
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

from generate_features import prepare_model_input
from utils import prepare_dataloader, make_predictions
from model import HalfBlock, AttentionDTINN, LogCoshLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

### This script can be used to reproduce the following results: validation, independent testing and davis/dtitr. ###
### As a default, it reproduces the davis/dtitr results. ###

def test_loop(dataloader, model, loss_fn):
    model.eval()
    preds = torch.empty(0, dtype=torch.int64, device=device)
    ys = torch.empty(0, dtype=torch.int64, device=device)
    with torch.no_grad():
        for (comp_x_cat, comp_x_cont, prot_x_cat, prot_x_cont, y) in dataloader:
            pred = model(comp_x_cat, comp_x_cont, prot_x_cat, prot_x_cont)
            ys = torch.concat([ys, y])
            preds = torch.concat([preds, pred])
      
    ys = ys.flatten()     
    preds = preds.flatten()
    
    test_loss = loss_fn(preds, ys)
    
    test_loss = float(test_loss.detach().cpu().numpy())
    ys_all_out = ys.detach().cpu().numpy()
    preds_all_out = preds.detach().cpu().numpy()
    spear = spearmanr(preds_all_out, ys_all_out).correlation
    rmse = mean_squared_error(ys_all_out, preds_all_out, squared=False)
    cindex = concordance_index(ys_all_out, preds_all_out)
    
    print("Test loss:", test_loss, "\nSpearman correlation:", spear,  "\nRMSE:", rmse, "\nC Index:", cindex)
    
    return preds_all_out, ys_all_out

def plot_correlation(ys, preds, interaction_score=False):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("ticks")
    fig, ax = plt.subplots(1, 2, sharey=True, figsize = (10,5), dpi=100)
    
    for i in range(2):
        if i == 0:
            ax[i].scatter(ys, preds, s=3, alpha=0.5, color="firebrick")
        else:
            sns.kdeplot(x=ys.flatten(), y=preds.flatten(), cmap=sns.light_palette('firebrick', as_cmap=True), fill=True, ax=ax[1])
            
        if interaction_score:
            ax[i].xaxis.set_ticks(np.arange(0, 1.1, 0.5))
            ax[i].yaxis.set_ticks(np.arange(0, 1.1, 0.5))
            straight = np.linspace(0, 1, num=100)
        else:
            ax[i].xaxis.set_ticks(np.arange(5, 10.1, 0.5))
            ax[i].yaxis.set_ticks(np.arange(5, 10.1, 0.5))
            straight = np.linspace(5, 10, num=100)
            
        ax[i].plot(straight, straight, color="gray", alpha=0.5, linewidth=2.4)
        ax[i].tick_params(axis='both', which='major', labelsize=15)
        ax[i].tick_params(axis='both', which='minor', labelsize=15)
        
    
    fig.supxlabel("True label", fontsize=18)
    fig.supylabel("Predicted label", fontsize=18)
    
    
    fig.patch.set_facecolor('white')
    sns.despine()
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--protein_class", action="store", help="Provide protein class to be tested.", default="kinase_davis_dtitr")
    parser.add_argument("-i", "--input_path", action="store", help="Provide test data input file path.", default="../data/davis_dtitr_data/final_test_data_filtered_zernike.csv")
    parser.add_argument("-j", "--jsonfile_path", action="store", help="Provide model config params.json file path.", default="../json_files/model_config_params.json")
    parser.add_argument("-l", "--label_type", action="store", help="Provide 'pchembl' or 'interaction_score' label type.", default="pchembl")
    parser.add_argument("-m", "--model_path", action="store", help="Provide model directory path to the directory containing 'pchembl_models' and 'interaction_score_models' directories.", default="../")
    parser.add_argument("-o", "--plot", action="store", help="Flag for plotting.", default=False)
    args = parser.parse_args()
    df_test = pd.read_csv(args.input_path).dropna()
    
    with open(args["jsonfile_path"], 'r') as json_file: # config for model architecture & hyperparams
        cfg_all = json.load(json_file)
    config = cfg_all[args.protein_class]
    if args.protein_class == "kinase_davis_dtitr":
        df_x = df_test.drop(["pchembl_value"], axis=1)
    else:
        df_x = df_test.drop(["pchembl_value", "interaction_strength"], axis=1)
    df_y = df_test.pchembl_value if args.label_type=="pchembl" else df_test.interaction_strength
    dataloader, shapes = prepare_dataloader(df_x, df_y, config)
    
    loss_fn = LogCoshLoss()
    model_comp = HalfBlock(shapes[0], shapes[1], config, "compound").to(device)
    model_prot = HalfBlock(shapes[2], shapes[3], config, "protein").to(device)
    model_combined = AttentionDTINN(config, model_comp, model_prot).to(device)
    all_preds = []
    
    weights_dir = f"{args.model_path}/{args.label_type}_models/{args.protein_class}"
    for i in range(config["n_ensemble"]):
        with open(f"{weights_dir}/model_{i}/dict_checkpoint.pkl", "rb") as f:
            checkpoint = pickle.load(f)
        model_combined.load_state_dict(checkpoint["model_weights"])
        preds, ys = test_loop(dataloader, model_combined, loss_fn)
        all_preds.append(preds)
        print("----------------------")

    ensemble_pred = np.mean(all_preds, axis=0)
    if config["n_ensemble"] > 1:
        print()
        print("Final ensemble prediction results:")
        print()
        print("Spearman correlation:", spearmanr(ensemble_pred, ys).correlation)
        print("RMSE:", mean_squared_error(ys, ensemble_pred, squared=False))
        print("C Index:", concordance_index(ys, ensemble_pred))
    
    
    if args.plot:
        plot_correlation(ys, ensemble_pred)

    
main()
