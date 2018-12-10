# Embedding-driven Relationship Alignment for Large Knowledge Graphs

This is our repository containing all tools and scripts used for:

- Creating custom benchmarks
- Training Knowledge Embeddings of benchmarks
- Detecting synonymous relations with such embeddings
- Evaluating detected synonyms

## Dependencies

- [OpenKE](https://github.com/thunlp/OpenKE) (compile library under thirdParty/OpenKE/ first!)
- Python3
- Bash (for some scripts)

## Files

### embedding.py

This python module wraps an embedding into a single class.

Example usage:
```
from thirdParty.OpenKE import models
from embedding import Embedding

emb = Embedding(benchmark, embedding, models.MODEL, embedding_dimensions=100)
```

This class provides access to methods like lookup\_{entity, relation} to lookup an entity or relation by its id.
Furthermore, it provides access to the embedding parameters.

### train\_embedding.py

This script starts a training for a given benchmark and specified model type.

Example usage:
```
python3 -m train_embedding --epoch-count 1000 --batch-count 10 transh benchmarks/FB15K/ embeddings/FB15K_transh/
```

This will load the benchmark located at *benchmarks/FB15K/* into OpenKE and will start training with model type **TransH**.
The resulting embedding is saved in this directory: *embeddings/FB15K_transh/*

For more options, take a look at:
```
python3 -m train_embedding -h
```

### train\_fb15k\_by\_model

This Bash-script will simply train multiple FB15K benchmarks.

### synonym\_analysis.py

This script will start the detection of synonymous relations.

Example usage:
```
python3 -m synonym_analysis -g transh benchmarks/FB15K/ embeddings/FB15K_transh/ experiments/FB15K_transh/
```

This will load the **TransH** knowledge embedding located in *embeddings/FB15K_transh/* and its corresponding benchmark in *benchmarks/FB15K/*.
Every output of the detection will be stored in *experiments/FB15K_transh/*.
Additionally, with the `-g` option, ground-truth data of synonymous relations in the specified benchmark will be loaded (if available) and precision-recall diagrams will be plotted.

For more options, take a look at:
```
python3 -m synonym_analysis -h
```

### analyse\_fb15k\_by\_model

This Bash-script will simply analyse multiple FB15K embeddings.

### plot\_evaluation.py

This script is used to plot precision-recall diagrams summarizing precision and recall for every model of an embedding.

Example usage:
```
python3 -m plot_evaluation -e experiments/FB15K
```

This will look for all experiment directories of FB15K (i.e. *experiments/FB15K\_transe/*, *experiments/FB15K\_transh/*, ...) and summarize their precision-recall plots into *experiments/FB15K_l1.pdf* and *experiments/FB15K_cos.pdf*.

### select\_gpu

This Bash-script will select the GPU(s) to use for training.

