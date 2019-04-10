#!/bin/bash

# ANALOGY
python3 -m synonym_analysis analogy benchmarks/dbpedia-201610N-1k-filtered embeddings/dbpedia-201610N-1k-filtered_analogy experiments/dbpedia-201610N-1k-filtered_analogy

# COMPLEX
python3 -m synonym_analysis complex benchmarks/dbpedia-201610N-1k-filtered embeddings/dbpedia-201610N-1k-filtered_complex experiments/dbpedia-201610N-1k-filtered_complex

# HOLE
python3 -m synonym_analysis hole benchmarks/dbpedia-201610N-1k-filtered embeddings/dbpedia-201610N-1k-filtered_hole experiments/dbpedia-201610N-1k-filtered_hole

# DISTMULT
python3 -m synonym_analysis distmult benchmarks/dbpedia-201610N-1k-filtered embeddings/dbpedia-201610N-1k-filtered_distmult experiments/dbpedia-201610N-1k-filtered_distmult

# RESCAL
python3 -m synonym_analysis rescal benchmarks/dbpedia-201610N-1k-filtered embeddings/dbpedia-201610N-1k-filtered_rescal experiments/dbpedia-201610N-1k-filtered_rescal

# TRANSE
python3 -m synonym_analysis transe benchmarks/dbpedia-201610N-1k-filtered embeddings/dbpedia-201610N-1k-filtered_transe experiments/dbpedia-201610N-1k-filtered_transe

# TRANSH
python3 -m synonym_analysis transh benchmarks/dbpedia-201610N-1k-filtered embeddings/dbpedia-201610N-1k-filtered_transh experiments/dbpedia-201610N-1k-filtered_transh

# TRANSD
python3 -m synonym_analysis transd benchmarks/dbpedia-201610N-1k-filtered embeddings/dbpedia-201610N-1k-filtered_transd experiments/dbpedia-201610N-1k-filtered_transd

