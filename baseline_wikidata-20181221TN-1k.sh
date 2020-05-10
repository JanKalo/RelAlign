#!/bin/bash

# wikidata-20181221TN-1k_2000_50
mkdir experiments/wikidata-20181221TN-1k_2000_50_baseline
python2 -m baseline benchmarks/wikidata-20181221TN-1k_2000_50/train2id.txt experiments/wikidata-20181221TN-1k_2000_50_baseline/synonyms_minSup_0.0001.txt 0.0001
python3 -m baseline_evaluation experiments/wikidata-20181221TN-1k_2000_50_baseline/synonyms_minSup_0.0001.txt benchmarks/wikidata-20181221TN-1k_2000_50/synonyms_id.txt

# PLOTTING
python3 -m plot_evaluation -e experiments/wikidata-20181221TN-1k_2000_50 -b experiments/wikidata-20181221TN-1k_2000_50_baseline/synonyms_minSup_0.0001_evaluation.txt

