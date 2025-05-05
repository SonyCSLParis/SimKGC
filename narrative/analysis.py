# -*- coding: utf-8 -*-
""" Analysis of results """
import os
import re
import ast
import click
from tqdm import tqdm
import pandas as pd

PATTERN = r'kg_base_prop_(?P<prop>[^_]+)_subevent_(?P<subevent>[^_]+)_role_(?P<role>[^_]+)_causation_(?P<causation>[^_]+)_syntax_simple_rdf_(?P<syntax>[^_]+)_lr(?P<lr>[^_]+)_bs(?P<batch_size>[^_]+)_ep(?P<epoch>\d+)'

def preprocess(x, k):
    if x == 'syntax':
        return k
    if x == 'lr':
        return float(k)
    return int(k)

def read_json_metric(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = {}
    print(lines)
    for l in lines:
        tm = l.split(": ")[0]
        val = ast.literal_eval(l.replace(f"{tm}: ", "").replace("\n", ""))
        data[tm] = val
    return data

def get_data_exp(folder, exp):
    params = re.match(PATTERN, exp).groupdict()
    params = {x: preprocess(x, k) for x, k in params.items()}
    finished = os.path.exists(os.path.join(folder, exp, 'model_best.mdl'))
    params.update({'finished': finished, "exp": exp})

    eval_f = "metrics_test.txt.json_model_best.mdl.json"
    has_eval = os.path.exists(os.path.join(folder, exp, eval_f))
    if has_eval:
        metrics = read_json_metric(os.path.join(folder, exp, eval_f))
        params.update(metrics['average metrics'])
    
    return params

def get_df_params(folder):
    exps = [x for x in os.listdir(folder) if x.startswith("kg_base_prop")]
    exps = [x for x in exps if "role_0" in x]
    data = []
    for exp in tqdm(exps):
        params = get_data_exp(folder, exp)
        data.append(params)

    df = pd.DataFrame(data)
    df["syntax"] = "simple_rdf_" + df["syntax"]
    return df
    
@click.command()
@click.option("--folder-in", default="narrative/experiments", help="Folder containing the experiment logs")
@click.option("--folder-out", default="narrative/results", help="Folder to store results")
def main(folder_in, folder_out):
    df = get_df_params(folder_in)
    print(df.columns)
    for x in ["prop", "subevent", "role", "causation"]:
        print(df.groupby(x).agg({"exp": "count"}))
    for x in ["lr", "epoch", "batch_size"]:
        print(df.groupby(x).agg({"exp": "count"}))
    df.to_csv(os.path.join(folder_out, "results.csv"))
    

if __name__ == "__main__":
    main()
