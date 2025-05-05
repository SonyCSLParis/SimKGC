# -*- coding: utf-8 -*-
""" Eval SimKGC experiments """
import os
import click
import subprocess
from loguru import logger
from tqdm import tqdm

DATA_F = os.path.expanduser("~/data/SimKGC/NarrativeInductiveDataset")
EVAL_COMMAND = """
python -u evaluate.py --task FB15k237 \
    --is-test --eval-model-path {} --neighbor-weight 0 \
        --rerank-n-hop 5 --train-path {} \
            --valid-path {}
"""
FILE_N = "metrics_test.txt.json_model_best.mdl.json"

def get_exps(folder):
    exps = [x for x in os.listdir(folder) if x.startswith("kg_base_prop")]
    exps = [x for x in exps if "role_0" in x]
    return [x for x in exps if os.path.exists(os.path.join(folder, x, 'model_best.mdl'))]


@click.command()
@click.option('--folder', default='narrative/experiments', type=str, help='Folder containing the experiments')
def main(folder):
    exps = get_exps(folder)
    for exp in tqdm(exps):
        if not os.path.exists(os.path.join(folder, exp, FILE_N)):
            logger.info(f"Evaluating EXP: {exp}")
            model_p = os.path.join(folder, exp, "model_best.mdl")
            train_p = os.path.join(DATA_F, exp.split('_lr')[0], 'train.txt.json')
            test_p = os.path.join(DATA_F, exp.split('_lr')[0], 'test.txt.json')
            command = EVAL_COMMAND.format(model_p, train_p, test_p)
            logger.info(f"Running command: {command}")
            subprocess.run(command, shell=True, check=False)
        else:
            logger.info(f"Already evaluated EXP: {exp}, skipping")



if __name__ == "__main__":
    main()

