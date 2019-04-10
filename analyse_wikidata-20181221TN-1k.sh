#!/bin/bash

# ANALOGY
python3 -m synonym_analysis -g analogy benchmarks/wikidata-20181221TN-1k_2000_50 embeddings/wikidata-20181221TN-1k_2000_50_analogy experiments/wikidata-20181221TN-1k_2000_50_analogy

# COMPLEX
python3 -m synonym_analysis -g complex benchmarks/wikidata-20181221TN-1k_2000_50 embeddings/wikidata-20181221TN-1k_2000_50_complex experiments/wikidata-20181221TN-1k_2000_50_complex

# HOLE
python3 -m synonym_analysis -g hole benchmarks/wikidata-20181221TN-1k_2000_50 embeddings/wikidata-20181221TN-1k_2000_50_hole experiments/wikidata-20181221TN-1k_2000_50_hole

# DISTMULT
python3 -m synonym_analysis -g distmult benchmarks/wikidata-20181221TN-1k_2000_50 embeddings/wikidata-20181221TN-1k_2000_50_distmult experiments/wikidata-20181221TN-1k_2000_50_distmult

# RESCAL
python3 -m synonym_analysis -g rescal benchmarks/wikidata-20181221TN-1k_2000_50 embeddings/wikidata-20181221TN-1k_2000_50_rescal experiments/wikidata-20181221TN-1k_2000_50_rescal

# TRANSE
python3 -m synonym_analysis -g transe benchmarks/wikidata-20181221TN-1k_2000_50 embeddings/wikidata-20181221TN-1k_2000_50_transe experiments/wikidata-20181221TN-1k_2000_50_transe

# TRANSH
python3 -m synonym_analysis -g transh benchmarks/wikidata-20181221TN-1k_2000_50 embeddings/wikidata-20181221TN-1k_2000_50_transh experiments/wikidata-20181221TN-1k_2000_50_transh

# TRANSD
python3 -m synonym_analysis -g transd benchmarks/wikidata-20181221TN-1k_2000_50 embeddings/wikidata-20181221TN-1k_2000_50_transd experiments/wikidata-20181221TN-1k_2000_50_transd

