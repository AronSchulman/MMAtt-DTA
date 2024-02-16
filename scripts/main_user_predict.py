import pandas as pd
import numpy as np
import torch
import json
import pickle

from generate_features import prepare_model_input
from utils import prepare_dataloader, make_predictions
from model import HalfBlock, AttentionDTINN

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():

    df_user_input = pd.read_csv("scrap_data/user_input.csv").dropna()
    
    cols = list(df_user_input.columns)
    cols.append("prediction")
    df_final = pd.DataFrame(columns=cols).dropna(axis=1, how='all') # dropna to avert warning of concatenation to empty dataframes
    
    with open('json_files/model_config_params.json', 'r') as json_file: # config for model architecture & hyperparams
        cfg_all = json.load(json_file)
    
    for prot in df_user_input.protein_class.unique(): # process by protein class
        df_user_per_prot = df_user_input[df_user_input.protein_class==prot]
        cfg = cfg_all[prot]
        for mod_type in df_user_per_prot.model_type.unique():# process by model type
            df_per_prot_and_mod = df_user_per_prot[df_user_per_prot.model_type==mod_type].copy()
            for i in range(len(df_per_prot_and_mod)): # process one line at a time - could be made more efficient!
                input_line = df_per_prot_and_mod.iloc[i]
                if i == 0:
                    selected_features = prepare_model_input(input_line.smiles, input_line.uniprot_id, input_line.protein_class)
                else:
                    selected_features = pd.concat([selected_features, prepare_model_input(input_line.smiles, input_line.uniprot_id, input_line.protein_class)])
    
    
            dataloader, shapes = prepare_dataloader(selected_features, pd.Series(selected_features.index), cfg)
    
            model_comp = HalfBlock(shapes[0], shapes[1], cfg, "compound").to(device)
            model_prot = HalfBlock(shapes[2], shapes[3], cfg, "protein").to(device)
            model_combined = AttentionDTINN(cfg, model_comp, model_prot).to(device)
    
            weights_dir = f"models/{input_line.model_type}_models/{input_line.protein_class}"
            ensemble_preds = []
            for i in range(5):
                with open(f"{weights_dir}/model_{i}/dict_checkpoint.pkl", "rb") as f:
                    checkpoint = pickle.load(f)
                model_combined.load_state_dict(checkpoint["model_weights"])
                preds = make_predictions(dataloader, model_combined)
                ensemble_preds.append(preds)
            mean_preds = np.mean(ensemble_preds, axis=0)
            df_per_prot_and_mod["prediction"] = mean_preds
            df_final = pd.concat([df_final, df_per_prot_and_mod])
    
    df_final = df_final.sort_index()
    df_final.to_csv("model_output_predictions.csv", index=False)
    
main()
