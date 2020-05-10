#!/bin/bash

# FB15K_2000_50
mkdir experiments/FB15K_2000_50_baseline
python2 -m baseline benchmarks/FB15K_2000_50/train2id.txt experiments/FB15K_2000_50_baseline/synonyms_minSup_0.02.txt 0.02
python3 -m baseline_evaluation experiments/FB15K_2000_50_baseline/synonyms_minSup_0.02.txt benchmarks/FB15K_2000_50/synonyms_id.txt

# PLOTTING
python3 -m plot_evaluation -e experiments/FB15K_2000_50 -b experiments/FB15K_2000_50_baseline/synonyms_minSup_0.02_evaluation.txt

