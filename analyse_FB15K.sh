#!/bin/bash

# ANALOGY
python3 -m synonym_analysis -g analogy benchmarks/FB15K_2000_50 embeddings/FB15K_2000_50_analogy experiments/FB15K_2000_50_analogy

# COMPLEX
python3 -m synonym_analysis -g complex benchmarks/FB15K_2000_50 embeddings/FB15K_2000_50_complex experiments/FB15K_2000_50_complex

# HOLE
python3 -m synonym_analysis -g hole benchmarks/FB15K_2000_50 embeddings/FB15K_2000_50_hole experiments/FB15K_2000_50_hole

# DISTMULT
python3 -m synonym_analysis -g distmult benchmarks/FB15K_2000_50 embeddings/FB15K_2000_50_distmult experiments/FB15K_2000_50_distmult

# RESCAL
python3 -m synonym_analysis -g rescal benchmarks/FB15K_2000_50 embeddings/FB15K_2000_50_rescal experiments/FB15K_2000_50_rescal

# TRANSE
python3 -m synonym_analysis -g transe benchmarks/FB15K_2000_50 embeddings/FB15K_2000_50_transe experiments/FB15K_2000_50_transe

# TRANSH
python3 -m synonym_analysis -g transh benchmarks/FB15K_2000_50 embeddings/FB15K_2000_50_transh experiments/FB15K_2000_50_transh

# TRANSD
python3 -m synonym_analysis -g transd benchmarks/FB15K_2000_50 embeddings/FB15K_2000_50_transd experiments/FB15K_2000_50_transd

