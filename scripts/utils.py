import torch
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

class DrugTargetDataset(Dataset):
    def __init__(self, comp_x_cat, comp_x_cont, prot_x_cat, prot_x_cont, labels):
        self.comp_x_cat = torch.tensor(comp_x_cat.values, dtype=torch.int64, device=device)
        self.prot_x_cat = torch.tensor(prot_x_cat.values, dtype=torch.int64, device=device)
        
        self.comp_x_cont = torch.tensor(comp_x_cont.values, dtype=torch.float32, device=device)
        self.prot_x_cont = torch.tensor(prot_x_cont.values, dtype=torch.float32, device=device)
        
        self.labels = torch.tensor(labels.values, dtype=torch.float32, device=device).reshape(-1,1)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        comp_cat_feats = self.comp_x_cat[idx]
        comp_cont_feats = self.comp_x_cont[idx]
        prot_cat_feats = self.prot_x_cat[idx]
        prot_cont_feats = self.prot_x_cont[idx]
        label = self.labels[idx]

        return comp_cat_feats, comp_cont_feats, prot_cat_feats, prot_cont_feats, label
    
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
    
    shapes = [df_comp_x_cont.shape[1], df_comp_x_cat.shape[1], df_prot_x_cont.shape[1], df_prot_x_cat.shape[1]] # return for building HalfBlocks
    
    return data_loader, shapes
    
def make_predictions(dataloader, model):
    model.eval()
    preds = torch.empty(0, dtype=torch.int64, device=device)
    with torch.no_grad():
        for (comp_x_cat, comp_x_cont, prot_x_cat, prot_x_cont, y) in dataloader:
            pred = model(comp_x_cat, comp_x_cont, prot_x_cat, prot_x_cont)
            preds = torch.concat([preds, pred])
            
    preds = preds.flatten()
    preds = preds.detach().cpu().numpy()

    return preds
    