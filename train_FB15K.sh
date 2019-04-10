#!/bin/bash

# ANALOGY
python3 -um train_embedding -c 500 -f 4 -l 0.2 -o Adagrad analogy benchmarks/FB15K_2000_50 embeddings/FB15K_2000_50_analogy | tee FB15K_2000_50_analogy.log

# COMPLEX
python3 -um train_embedding -c 500 -f 4 -l 0.2 -o Adagrad complex benchmarks/FB15K_2000_50 embeddings/FB15K_2000_50_complex | tee FB15K_2000_50_complex.log

# HOLE
python3 -um train_embedding -c 500 -f 4 -l 0.2 -o Adagrad hole benchmarks/FB15K_2000_50 embeddings/FB15K_2000_50_hole | tee FB15K_2000_50_hole.log

# DISTMULT
python3 -um train_embedding -c 500 -f 4 -l 0.2 -o Adagrad distmult benchmarks/FB15K_2000_50 embeddings/FB15K_2000_50_distmult | tee FB15K_2000_50_distmult.log

# RESCAL
python3 -um train_embedding -c 1000 -f 50 -l 0.001 rescal benchmarks/FB15K_2000_50 embeddings/FB15K_2000_50_rescal | tee FB15K_2000_50_rescal.log

# TRANSE
python3 -um train_embedding -c 2000 -f 1 -l 0.00001 transe benchmarks/FB15K_2000_50 embeddings/FB15K_2000_50_transe | tee FB15K_2000_50_transe.log

# TRANSH
python3 -um train_embedding -c 2000 -f 10 -l 0.00001 transh benchmarks/FB15K_2000_50 embeddings/FB15K_2000_50_transh | tee FB15K_2000_50_transh.log

# TRANSD
python3 -um train_embedding -c 2000 -f 10 -l 0.0001 transd benchmarks/FB15K_2000_50 embeddings/FB15K_2000_50_transd | tee FB15K_2000_50_transd.log

