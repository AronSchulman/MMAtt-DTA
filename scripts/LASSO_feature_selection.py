import argparse
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoCV
from joblib import parallel_backend

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jsonfile_path", action="store", help="Provide params.json file path.", default="../json_files/LASSO_feature_selection_params.json")
    args = parser.parse_args()
    try:
        with open(args.jsonfile_path, 'r') as json_file:
            cfg = json.load(json_file)

        proteins = cfg["proteins"]
        for prot in proteins:
            try:
                df = pd.read_csv(f"{cfg['input_path']}/{prot}_lasso_input.csv")
            
        
                ### run LASSO n_iterations times and take top n_features_to_select for each protein family ###
                ### the top features is determined by how many iterations would select the feature; more the better ###
                ### e.g. a feature might be selected by 92 LASSOs out of 100, and another only 46 times ###
                
                ### The y label should be in column called "interaction_strength". Columns that are kept outside the selection process are ###
                ### in this case appended with "_protein_categorical", referring to protein subclass labels. ###
            
                force_keep_cols = ["interaction_strength"] + [col for col in df.columns if 'protein_categorical' in col] # keep interaction strength & potential subclass labels
                df_x = df.drop(force_keep_cols, axis=1)
                df_y = df["interaction_strength"]
                importance_boolean_matrix = np.zeros((cfg["n_iterations"], df_x.shape[1])) # for stability selection
            
                for i in range(cfg["n_iterations"]):
                    with parallel_backend('threading', n_jobs=-1):
                        lasso_model = make_pipeline(StandardScaler(), LassoCV(cv=cfg["num_folds"], tol=cfg["tolerance"], verbose=cfg["verbose"], n_jobs=-1, selection='random')).fit(df_x, df_y)
            
                    importance = np.abs(lasso_model[1].coef_)
                    importance_boolean = []
                    for x in importance:
                        if x == 0:
                            importance_boolean.append(False)
                        else:
                            importance_boolean.append(True)
            
                    importance_boolean_matrix[i] = importance_boolean
            
                zero_counts = np.count_nonzero(importance_boolean_matrix, axis=0)
                arrinds = zero_counts.argsort()
                zero_ordered = zero_counts[arrinds[::-1]]
                col_ordered = np.array(df_x.columns)[arrinds[::-1]]
                cols_to_choose = list(col_ordered[0:cfg["n_features_to_select"]])
                final_cols = force_keep_cols + cols_to_choose
                df_x_selected = df[final_cols]
                df_final = pd.Series(df_x_selected.columns) # only column names are needed
            
                save_name = f"{prot}_selected_feature_names.csv"
                df_final.to_csv(f"{cfg['save_path']}/{save_name}", index=False)
                
            except FileNotFoundError:
                print("Could not open file", f"{cfg['input_path']}/{prot}_lasso_input.csv")
    except FileNotFoundError:
        print(f"Could not open file {args.file_path}")

main()


