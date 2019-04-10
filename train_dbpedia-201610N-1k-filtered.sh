#!/bin/bash

# ANALOGY
python3 -um train_embedding -c 500 -f 100 -l 0.1 -o Adagrad analogy benchmarks/dbpedia-201610N-1k-filtered embeddings/dbpedia-201610N-1k-filtered_analogy | tee dbpedia-201610N-1k-filtered_analogy.log

# COMPLEX
python3 -um train_embedding -c 500 -f 100 -l 0.1 -o Adagrad complex benchmarks/dbpedia-201610N-1k-filtered embeddings/dbpedia-201610N-1k-filtered_complex | tee dbpedia-201610N-1k-filtered_complex.log

# HOLE
python3 -um train_embedding -c 500 -f 50 -l 0.1 -o Adagrad hole benchmarks/dbpedia-201610N-1k-filtered embeddings/dbpedia-201610N-1k-filtered_hole | tee dbpedia-201610N-1k-filtered_hole.log

# DISTMULT
python3 -um train_embedding -c 500 -f 50 -l 0.1 -o Adagrad distmult benchmarks/dbpedia-201610N-1k-filtered embeddings/dbpedia-201610N-1k-filtered_distmult | tee dbpedia-201610N-1k-filtered_distmult.log

# RESCAL
python3 -um train_embedding -c 500 -f 1000 -l 0.001 rescal benchmarks/dbpedia-201610N-1k-filtered embeddings/dbpedia-201610N-1k-filtered_rescal | tee dbpedia-201610N-1k-filtered_rescal.log

# TRANSE
python3 -um train_embedding -c 500 -f 50 -l 0.0001 transe benchmarks/dbpedia-201610N-1k-filtered embeddings/dbpedia-201610N-1k-filtered_transe | tee dbpedia-201610N-1k-filtered_transe.log

# TRANSH
python3 -um train_embedding -c 500 -f 50 -l 0.0001 transh benchmarks/dbpedia-201610N-1k-filtered embeddings/dbpedia-201610N-1k-filtered_transh | tee dbpedia-201610N-1k-filtered_transh.log

# TRANSD
python3 -um train_embedding -c 500 -f 50 -l 0.0001 transd benchmarks/dbpedia-201610N-1k-filtered embeddings/dbpedia-201610N-1k-filtered_transd | tee dbpedia-201610N-1k-filtered_transd.log

