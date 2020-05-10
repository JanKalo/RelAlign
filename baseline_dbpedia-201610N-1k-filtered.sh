#!/bin/bash

# dbpedia-201610N-1k-filtered
mkdir experiments/dbpedia-201610N-1k-filtered_baseline
python2 -m baseline benchmarks/dbpedia-201610N-1k-filtered/train2id.txt experiments/dbpedia-201610N-1k-filtered_baseline/synonyms_minSup_0.001.txt 0.001

# EVALUATION & PLOTTING
python3 -m id2uri experiments/dbpedia-201610N-1k-filtered_baseline/synonyms_minSup_0.001.txt benchmarks/dbpedia-201610N-1k-filtered/relation2id.txt
python3 -m evaluate_dbpedia experiments/dbpedia-201610N-1k-filtered_combined_approx_500_correct.txt experiments/dbpedia-201610N-1k-filtered_baseline/synonyms_minSup_0.001_uris.txt

