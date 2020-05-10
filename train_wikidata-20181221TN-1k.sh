#!/bin/bash

# ANALOGY
python3 -um train_embedding -c 1000 -f 100 -l 0.1 -o Adagrad analogy benchmarks/wikidata-20181221TN-1k_2000_50 embeddings/wikidata-20181221TN-1k_2000_50_analogy | tee wikidata-20181221TN-1k_2000_50_analogy.log

# COMPLEX
python3 -um train_embedding -c 1000 -f 100 -l 0.1 -o Adagrad complex benchmarks/wikidata-20181221TN-1k_2000_50 embeddings/wikidata-20181221TN-1k_2000_50_complex | tee wikidata-20181221TN-1k_2000_50_complex.log

# HOLE
python3 -um train_embedding -c 1000 -f 50 -l 0.1 -o Adagrad hole benchmarks/wikidata-20181221TN-1k_2000_50 embeddings/wikidata-20181221TN-1k_2000_50_hole | tee wikidata-20181221TN-1k_2000_50_hole.log

# DISTMULT
python3 -um train_embedding -c 1000 -f 30 -l 0.1 -o Adagrad distmult benchmarks/wikidata-20181221TN-1k_2000_50 embeddings/wikidata-20181221TN-1k_2000_50_distmult | tee wikidata-20181221TN-1k_2000_50_distmult.log

# RESCAL
python3 -um train_embedding -c 1000 -f 1200 -l 0.001 rescal benchmarks/wikidata-20181221TN-1k_2000_50 embeddings/wikidata-20181221TN-1k_2000_50_rescal | tee wikidata-20181221TN-1k_2000_50_rescal.log

# TRANSE
python3 -um train_embedding -c 1000 -f 30 -l 0.001 transe benchmarks/wikidata-20181221TN-1k_2000_50 embeddings/wikidata-20181221TN-1k_2000_50_transe | tee wikidata-20181221TN-1k_2000_50_transe.log

# TRANSH
python3 -um train_embedding -c 1000 -f 250 -l 0.001 transh benchmarks/wikidata-20181221TN-1k_2000_50 embeddings/wikidata-20181221TN-1k_2000_50_transh | tee wikidata-20181221TN-1k_2000_50_transh.log

# TRANSD
python3 -um train_embedding -c 1000 -f 200 -l 0.001 transd benchmarks/wikidata-20181221TN-1k_2000_50 embeddings/wikidata-20181221TN-1k_2000_50_transd | tee wikidata-20181221TN-1k_2000_50_transd.log

