# -*- coding: utf-8 -*-
""" Run SimKGC experiments """
import os
import subprocess
import multiprocessing

# DS VERSIONS
VP = os.path.expanduser("~/data/SimKGC/NarrativeInductiveDataset")
VERSIONS = sorted(os.listdir(VP))

LEARNING_RATES = [1e-5, 3e-5, 5e-5]
EPOCHS = [1, 10, 50]
BATCH_SIZE = [256, 512, 1024]

OUTPUT_F = "./narrative/experiments"
if not os.path.exists(OUTPUT_F):
    os.makedirs(OUTPUT_F)

def main():
    for lr in LEARNING_RATES:
        for epoch in EPOCHS:
            for bs in BATCH_SIZE:
                for v in VERSIONS:
                    name = f"{v}_lr{lr}_bs{bs}_ep{epoch}"
                    output_dir = os.path.join(OUTPUT_F, name)
                    if not os.path.exists(os.path.join(output_dir, "model_best.mdl")):
                        os.makedirs(output_dir, exist_ok=True)

                        data_dir = os.path.join(VP, v)
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
                        --batch-size {bs} \
                        --print-freq 20 \
                        --additive-margin 0.02 \
                        --use-amp \
                        --use-self-negative \
                        --finetune-t \
                        --pre-batch 2 \
                        --epochs {epoch} \
                        --workers {workers} \
                        --max-to-keep 5 "$@"
                        """
                        subprocess.run(command, shell=True, check=True)
