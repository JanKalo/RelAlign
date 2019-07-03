# OpenKE dataset directory

This directory contains the OpenKE datasets and two additional scripts to generate the input files.
For each dataset a directory containing the following files is needed:

- entity2id.txt (OID mapped entities)
- relation2id.txt (OID mapped relations)
- train2id.txt (OID triples \<e1, e2, r\>)
- valid2id.txt (OID triples for validation, optional)
- test2id.txt (OID triples for testing, optional)

## map\_input.py

This script maps a flat triple file (i.e. \*.nt file) into an OpenKE readable format.
It should work with Python 2 and Python 3.

Example usage for generating a OpenKE readable files with validation and test triples
(80% for training, 10% each for validation and test):

```shell
$ python -m map_input flat_triple_file.nt 1
```

Example usage for generating a dataset without validation and test triples:

```shell
$ python -m map_input flat_triple_file.nt 0
```

The dataset files will be stored in the same directory from where the script was executed.

## synonym\_inject.py

This script created synthetic synonyms for an existing dataset.

Usage:

```shell
$ python3 -m synonym_inject -h
usage: synonym_inject.py [-h] [-p PERCENTAGE_PER_RELATION]
                         [-o MIN_RELATION_OCCURENCE]
                         [-f {inject_synonym_1,inject_synonym_2}]
                         INPUT_BENCHMARK

positional arguments:
    INPUT_BENCHMARK       Input Benchmark to inject synonyms to.

optional arguments:
    -h, --help            show this help message and exit
    -p PERCENTAGE_PER_RELATION, --percentage-per-relation PERCENTAGE_PER_RELATION
                          Percentage in range [0, 1] of relations to replace
                          with one new artificial synonym relation (Default: 0.5).
    -o MIN_RELATION_OCCURENCE, --min-relation-occurence MIN_RELATION_OCCURENCE
                          Only injecting artificial synonym relations for
                          relations which occur at least this often in its
                          benchmark (Default: 1000).
    -f {inject_synonym_1,inject_synonym_2}, --func_inject_synonym {inject_synonym_1,inject_synonym_2}
                          The injection func to use (Default: inject_synonym_1)
```

`INPUT_BENCHMARK` is the directory of the target dataset.
The new dataset will be saved in the same directory where `INPUT_BENCHMARK` is located.

## Our datasets

Because of the size of our datasets, we hosted the datasets externally.
