# Knowledge Graph Consolidation by Unifying Synonymous Relationships

If you use the code, please cite the following paper:

```
@inproceedings{kalo2019iswc,
  title={Knowledge Graph Consolidation by Unifying Synonymous Relationships},
  author={Kalo, Jan-Christoph and Ehler, Philipp and Balke, Wolf-Tilo},
  booktitle="Proceedings of the 18th International Semantic Web Conference on The Semantic Web",
  series = {ISWC '19},
  pages={},
  year={2019}
}
```


This is our repository containing all tools and scripts used for:

- Creating input files from .nt files.
- Training knowledge embeddings using Tensorflow and OpenKE
- Detecting synonymous relations using the embeddings
- Evaluating detected synonyms and creating plots

## Dependencies

- [OpenKE](https://github.com/thunlp/OpenKE) (compile library under thirdParty/OpenKE/ first!)
- Python2 with pyfpgrowth and pyspark (for baseline)
- Python3
- Bash (for some scripts)

## Python Files

### embedding.py

This python module wraps an embedding into a single class.

Example usage:

```python
from thirdParty.OpenKE import models
from embedding import Embedding

emb = Embedding(benchmark, embedding, models.MODEL, embedding_dimensions=100)
```

This class provides access to methods like lookup\_{entity, relation} to lookup an entity or relation by its id.
Furthermore, it provides access to the embedding parameters.

### train\_embedding.py

This script starts a training for a given benchmark and specified model type.

Example usage:

```shell
$ python3 -m train_embedding --epoch-count 1000 --batch-count 10 transh benchmarks/FB15K/ embeddings/FB15K_transh/
```

This will load the dataset located at `benchmarks/FB15K/` into OpenKE and will start training with model type **TransH**.
The resulting embedding is saved in this directory: `embeddings/FB15K_transh/`

For more options, see:

```shell
$ python3 -m train_embedding -h
```

### synonym\_analysis.py

This script will start the detection of synonymous relations.

Example usage:

```shell
$ python3 -m synonym_analysis -g transh benchmarks/FB15K/ embeddings/FB15K_transh/ experiments/FB15K_transh/
```

This will load the **TransH** knowledge embedding located in `embeddings/FB15K_transh/` and its corresponding benchmark in `benchmarks/FB15K/`.
Every output of the detection will be stored in `experiments/FB15K_transh/`.
Additionally, with the `-g` option, ground-truth data of synonymous relations in the specified benchmark will be loaded (if available) and precision-recall diagrams will be plotted.

For more options, see:

```shell
$ python3 -m synonym_analysis -h
```

### baseline.py

This script contains the code for our baseline method to detect synonymous relationships.

Example usage:

```shell
$ python2 -m baseline benchmarks/FB15K_2000_50/train2id.txt experiments/FB15K_2000_50_baseline/synonyms_minSup_0.02.txt 0.02
```

The first parameter is the input triples file, the second parameter ist the output file and the third parameter specifies the minimum support value.

### baseline\_evaluation.py

This script calculates the precision-recall values for a given baseline.py output and gold-standard.

Example usage:

```shell
$ python3 -m baseline_evaluation experiments/FB15K_2000_50_baseline/synonyms_minSup_0.02.txt benchmarks/FB15K_2000_50/synonyms_id.txt
```

The output is placed in the directory where the baseline.py output is located.
(In the example above: `experiments/FB15K_2000_50_baseline/`)

For more options, see:

```shell
$ python3 -m baseline_evaluation -h
```

### plot\_evaluation.py

This script is used to plot precision-recall diagrams summarizing precision and recall for every model of an embedding.

Example usage:

```shell
$ python3 -m plot_evaluation -e experiments/FB15K
```

This will look for all experiment directories of FB15K (i.e. `experiments/FB15K_transe/`, `experiments/FB15K_transh/`, ...) and create respective precision-recall plots as `experiments/FB15K_l1.pdf` and `experiments/FB15K_cos.pdf`.

### evaluate\_dbpedia.py

This script is performing the evaluation of our manually evaluated DBpedia dataset (including precision@k diagrams) for a given (manually crafted) gold standard and a baseline.py output.
This script internally contains the relevant classification files for each model and similarity function we used.

Example usage:

```shell
$ python3 -m evaluate_dbpedia experiments/dbpedia-201610N-1k-filtered_combined_approx_500_correct.txt experiments/dbpedia-201610N-1k-filtered_baseline/synonyms_minSup_0.001_uris.txt
```

For more options, see:

```shell
$ python3 -m evaluate_dbpedia -h
```

### id2uri.py

This script takes as input a file with ID pairs and calculates the corresponding file with URI pairs by looking them up in a given relation2id.txt file.

Example usage:

```shell
$ python3 -m id2uri experiments/dbpedia-201610N-1k-filtered_baseline/synonyms_minSup_0.001.txt benchmarks/dbpedia-201610N-1k-filtered/relation2id.txt
```

The URI pairs file is saved in the same directory as the ID pairs file.

For more options, see:

```shell
$ python3 -m id2uri -h
```

## Bash Files

### select\_gpu

This Bash-script will select the GPU(s) to use for training.

### train\_\*.sh

Training scripts for our experiments. Trains embeddings for our benchmarks.

### analyse\_\*.sh

Performs synonym detection for our experiments with our method on each embedding.

### baseline\_\*.sh

Performs synonym detection for our experiments with our baseline method.
Also plots all results.

## Experiments

In this subsection, we describe how to reproduce the results described in the evaluation section of the paper.
First, we describe the synthetic synonym creation, followed by the training of the respective datasets.
Afterwards, we synonym detection is performed and the results are evaluated and plots in .pdf format are created.

### Datasets

Our Freebase, DBpedia and Wikidata samples are available under: https://doi.org/10.6084/m9.figshare.8490134
The manually evaluated baseline is available under: https://figshare.com/s/11d4af3169a0e6d2437b

### Synthetic Synonyms

To create synthetic synonyms, it is necessary to change into the `benchmarks` directory.

```shell
$ cd ./benchmarks/.
```

Here, we can access our existing benchmarks and the `synonym_inject.py` script.
The script always saves the created synonym pairs in a text file inside the new benchmark folder which we can use as a ground-truth baseline.

###### 5.1 Synthetic Synonyms in Freebase

The following command creates our copy of the original FB15K benchmark with synthetic synonyms, which randomly replaces the occurrences of the corresponding relations (occuring at least 2000 times) for 50% of the subjects.
This benchmark we called `FB15K_2000_50`.

```shell
$ python3 -m synonym_inject --percentage-per-relation 0.5 --min-relation-occurence 2000 --func_inject_synonym inject_synonym_2 FB15K
```

###### 5.2 Synthetic Synonyms in Wikidata

We also created a copy of our Wikidata sample with the same parameters.
This benchmark we called `wikidata-20181221TN-1k_2000_50`.

```shell
$ python3 -m synonym_inject --percentage-per-relation 0.5 --min-relation-occurence 2000 --func_inject_synonym inject_synonym_2 wikidata-20181221TN-1k
```

### Training

For the training, we have to change the working directory to the root of the repository (if not done yet) because we need the `train_embedding.py` script.

```shell
$ cd ./
```

For all embeddings, we mainly tweaked the following parameters:

- `epoch-count`
- `batch-count`
- `learning-rate`

Apart from that, we always used 100 dimensions and generally sticked to the default values for all other parameters (see `python3 -m train_embedding -h`).

For more information, see the corresponding Bash scripts.

###### 5.1 FB15K Datasets

```shell
$ ./train_FB15K.sh
```

###### 5.2 Wikidata Datasets

```shell
$ ./train_wikidata-20181221TN-1k.sh
```

###### 5.3 DBpedia Dataset

```shell
$ ./train_dbpedia-201610N-1k-filtered.sh
```

### Synonym Detection and Evaluation with our method

Again, we need to change to the root of the repository, if not done yet.
Additionally, it is a good idea to prevent tensorflow-gpu to load the embeddings into the GPU VRAM because we don't want to train anything.

```shell
$ cd ./
$ . ./select_gpu -2
```

For the `synonym_analysis.py` script, note that we have to specify the correct number of embedding dimensions.

For more information, see the corresponding Bash scripts.

###### 5.1 FB15K Datasets

```shell
$ ./analyse_FB15K.sh
```

###### 5.2 Wikidata Datasets

```shell
$ ./analyse_wikidata-20181221TN-1k.sh
```

###### 5.3 DBpedia Dataset

Here, we excluded the `--ground-truth-available` option, because the synonym pairs are unknown.
Thus, we had to manually evaluate the classified synonym pairs.

```shell
$ ./analyse_dbpedia-201610N-1k-filtered.sh
```

### Synonym Detection and Evaluation with our baseline and Plotting of all results

Again, we need to change to the root of the repository if not done yet.

```shell
$ cd ./
```

The following scripts will perform the baseline evaluation and plot all results calculated up until now.
Note that the evaluation of our method is already performed in the analysis scripts in the previous section.
Because of the slightly different evaluation approach in our DBpedia experiment, the overall evaluation is performed _with_ the baseline.py output in this section.

For more information, see the corresponding Bash scripts.

###### 5.1 FB15K Datasets

```shell
$ ./baseline_FB15K.sh
```

###### 5.2 Wikidata Datasets

```shell
$ ./baseline_wikidata-20181221TN-1k.sh
```

###### 5.3 DBpedia Dataset

```shell
$ ./baseline_dbpedia-201610N-1k-filtered.sh
```

