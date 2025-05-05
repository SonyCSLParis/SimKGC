# -*- coding: utf-8 -*-
""" Train SimKGC experiments """
import os
import multiprocessing
import subprocess
import click

@click.command()
@click.option('data_dir')
@click.argument('output_f')
@click.option('--lr', default=1e-5, type=float)
@click.option('--batch_size', default=1024, type=int)
@click.option('--epochs', default=10, type=int)
def main(data_dir, output_f, lr, batch_size, epochs):
    version = data_dir.split('/')[-1]
    name = f"{version}_lr{lr}_bs{batch_size}_ep{epochs}"
    output_dir = os.path.join(output_f, name)

    train_path = os.path.join(data_dir, 'train.txt.json')
    valid_path = os.path.join(data_dir, 'valid.txt.json')
    workers = max(1, multiprocessing.cpu_count() - 4)

    command = f"""
    python3 -u main.py \
    --model-dir "{output_dir}" \
    --pretrained-model bert-base-uncased \
    --pooling mean \
    --lr {lr} \
    --use-link-graph \
    --train-path "{train_path}" \
    --valid-path "{valid_path}" \
    --task "FB15k237" \
    --batch-size {batch_size} \
    --print-freq 20 \
    --additive-margin 0.02 \
    --use-amp \
    --use-self-negative \
    --finetune-t \
    --pre-batch 2 \
    --epochs {epochs} \
    --workers {workers} \
    --max-to-keep 5 "$@"
    """
    subprocess.run(command, shell=True, check=False)