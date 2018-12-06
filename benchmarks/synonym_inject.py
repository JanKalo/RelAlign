import pandas as pd
import numpy as np
import random as rd
import os
from shutil import copyfile
from argparse import ArgumentParser

def get_df_mappings(fn_mappings):
    file_mappings = open(fn_mappings)
    list_mappings = map(lambda mapping: mapping.split(), file_mappings.readlines()[1:])
    list_mappings = map(lambda mapping: [mapping[0], int(mapping[1])], list_mappings)
    list_mappings = list(list_mappings)
    df_mappings = pd.DataFrame(list_mappings, columns = ["name", "idx"])
    return df_mappings

def get_df_triples(fn_triples):
    file_triples = open(fn_triples)
    list_triples = map(lambda triple: triple.split(), file_triples.readlines()[1:])
    list_triples = map(lambda triple: [int(triple[0]), int(triple[1]), int(triple[2])], list_triples)
    list_triples = list(list_triples)
    df_triples = pd.DataFrame(list_triples, columns = ["sub_idx", "obj_idx", "rel_idx"])
    return df_triples

def inject_synonym_1(df_relations, df_triples, rel_idx, percentage):
    # log
    print("----------------------------------------------------------------")
    print("INJECTING SYNONYM FOR RELATION %d" % rel_idx)
    
    # check
    rel_syn_idx = df_relations["idx"].max() + 1
    if rel_idx < 0 and rel_idx >= rel_syn_idx:
        print("relation index '%d' out of range" % rel_id)
        return
    
    # get relation series
    series_relation = df_relations.loc[df_relations["idx"] == rel_idx]
    if len(series_relation) != 1:
        print("something is wrong with relation mapping")
        return
    series_relation = series_relation.iloc[0]

    # get all triples with relation rel_idx and shuffle them
    df_rel_triples = df_triples.loc[df_triples["rel_idx"] == rel_idx]
    df_rel_triples = df_rel_triples.reindex(np.random.permutation(df_rel_triples.index))
    
    # get count how many occurences of the relation should be replaced
    # with an artificial synonym relation
    syn_count = int(len(df_rel_triples) * percentage)
    if syn_count <= 0:
        print("nothing to inject")
        return df_relations, df_triples, None
    print("injecting new synonym for relation '%s'" % series_relation["name"])
    print("... replacing %d occurences with new synonym relation from a total of %d occurences" \
            % (syn_count, len(df_rel_triples)))

    # track new synonym relation in df_relations
    rel_syn_name = "%s_synonym" % series_relation["name"]
    df_relations = df_relations.append(pd.DataFrame([[rel_syn_name, rel_syn_idx]], \
            columns = ["name", "idx"]), ignore_index=True)

    # now replace the occurences
    for i in range(syn_count):
        df_triples.iloc[df_rel_triples.index[i]]["rel_idx"] = rel_syn_idx

    # track dataframe with relation and its synonym
    df_synonym = pd.DataFrame([[series_relation["name"], rel_syn_name, rel_idx, rel_syn_idx]], \
            columns = ["rel_name", "rel_syn_name", "rel_idx", "rel_syn_idx"])

    # done
    return df_relations, df_triples, df_synonym

def inject_synonym_2(df_relations, df_triples, rel_idx, percentage):
    # log
    print("----------------------------------------------------------------")
    print("INJECTING SYNONYM FOR RELATION %d" % rel_idx)
    
    # check
    rel_syn_idx = df_relations["idx"].max() + 1
    if rel_idx < 0 and rel_idx >= rel_syn_idx:
        print("relation index '%d' out of range" % rel_id)
        return
    
    # get relation series
    series_relation = df_relations.loc[df_relations["idx"] == rel_idx]
    if len(series_relation) != 1:
        print("something is wrong with relation mapping")
        return
    series_relation = series_relation.iloc[0]

    # get all triples with relation rel_idx
    df_rel_triples = df_triples.loc[df_triples["rel_idx"] == rel_idx]
    
    # get list of distinct subject of these triples and shuffle them
    # to get the triples where we should add a synonym
    subjects = list(set(df_rel_triples["sub_idx"].values))
    rd.shuffle(subjects)
    subjects_count = int(len(subjects) * percentage)
    subjects = set(subjects[0:subjects_count])
    df_rel_triples = df_rel_triples.loc[df_rel_triples["sub_idx"].isin(subjects)]

    # get count how many occurences of the relation should be replaced
    # with an artificial synonym relation
    syn_count = len(df_rel_triples)
    if syn_count <= 0:
        print("nothing to inject")
        return df_relations, df_triples, None
    print("injecting new synonym for relation '%s'" % series_relation["name"])
    print("... replacing %d relations with new synonym relation for %d subjects" \
            % (syn_count, subjects_count))

    # track new synonym relation in df_relations
    rel_syn_name = "%s_synonym" % series_relation["name"]
    df_relations = df_relations.append(pd.DataFrame([[rel_syn_name, rel_syn_idx]], \
            columns = ["name", "idx"]), ignore_index=True)

    # now replace the occurences
    for i in range(syn_count):
        df_triples.iloc[df_rel_triples.index[i]]["rel_idx"] = rel_syn_idx

    # track dataframe with relation and its synonym
    df_synonym = pd.DataFrame([[series_relation["name"], rel_syn_name, rel_idx, rel_syn_idx]], \
            columns = ["rel_name", "rel_syn_name", "rel_idx", "rel_syn_idx"])

    # done
    return df_relations, df_triples, df_synonym

def inject_synonyms(df_relations, df_triples, percentage_per_relation, min_relation_occurence, \
        func_inject_synonym=inject_synonym_1):
    """
    inject synonym relations for random relations,
    which appear at least 'min_relation_occurence' times
    """
    # create copy of input dataframes
    df_relations_new = df_relations.copy(deep=True)
    df_triples_new = df_triples.copy(deep=True)

    # prepare synonyms dataframe with new synonyms
    df_synonyms = pd.DataFrame(columns = ["rel_name", "rel_syn_name", "rel_idx", "rel_syn_idx"])

    # begin injection
    rel_idx_max = df_relations_new["idx"].max()
    for rel_idx in range(rel_idx_max + 1):
        if rel_idx not in df_synonyms["rel_idx"].values \
                and len(df_triples_new.loc[df_triples_new["rel_idx"] == rel_idx]) \
                >= min_relation_occurence:
            df_relations_new, df_triples_new, df_synonym = func_inject_synonym(df_relations_new, \
                    df_triples_new, rel_idx, percentage_per_relation)
            if df_synonym is not None:
                df_synonyms = df_synonyms.append(df_synonym, ignore_index=True)
    
    # done
    return df_relations_new, df_triples_new, df_synonyms

def save_mappings(df_mappings, filename):
    file_mappings = open(filename, "w")
    row_mapper = lambda row: "%s\t%d\n" % (row[1]["name"], row[1]["idx"])
    file_mappings.write("%d\n" % len(df_mappings))
    file_mappings.writelines(map(row_mapper, df_mappings.iterrows()))
    file_mappings.close()

def save_triples(df_triples, filename):
    file_triples = open(filename, "w")
    row_mapper = lambda row: \
            "%d\t%d\t%d\n" % (row[1]["sub_idx"], row[1]["obj_idx"], row[1]["rel_idx"])
    file_triples.write("%d\n" % len(df_triples))
    file_triples.writelines(map(row_mapper, df_triples.iterrows()))
    file_triples.close()

def save_synonyms(df_synonyms, filename, write_names):
    file_synonyms = open(filename, "w")
    row_mapper_name = lambda row: "%s\t%s\n" % (row[1]["rel_name"], row[1]["rel_syn_name"])
    row_mapper_idx = lambda row: "%d\t%d\n" % (row[1]["rel_idx"], row[1]["rel_syn_idx"])
    file_synonyms.write("%d\n" % len(df_synonyms))
    if write_names:
        file_synonyms.writelines(map(row_mapper_name, df_synonyms.iterrows()))
    else:
        file_synonyms.writelines(map(row_mapper_idx, df_synonyms.iterrows()))
    file_synonyms.close()

def inject_benchmark(benchmark_dir, percentages_per_relation, min_relation_occurences, \
        func_inject_synonym=inject_synonym_1):
    # load relation mapping and triples
    df_relations = get_df_mappings(os.path.join(benchmark_dir, "relation2id.txt"))
    df_triples = get_df_triples(os.path.join(benchmark_dir, "train2id.txt"))

    # get directory information of benchmark
    stripped_dir = benchmark_dir.rstrip("/")
    dirname = os.path.dirname(stripped_dir)
    basename = os.path.basename(stripped_dir)

    # for each parameter
    for percentage in percentages_per_relation:
        for min_occurence in min_relation_occurences:
            # inject synonyms
            df_relations_new, df_triples_new, df_synonyms = \
                    inject_synonyms(df_relations, df_triples, percentage, min_occurence, \
                    func_inject_synonym)

            # save to new benchmark folder
            if func_inject_synonym == inject_synonym_1:
                new_benchmark_dir = os.path.join(dirname, \
                        "%s_%s_%s" % (basename, int(percentage * 100), min_occurence))
            else:
                new_benchmark_dir = os.path.join(dirname, \
                        "%s_%s_%s" % (basename, min_occurence, int(percentage * 100)))
            print("SAVING %s" % new_benchmark_dir)
            os.mkdir(new_benchmark_dir)
            save_mappings(df_relations_new, os.path.join(new_benchmark_dir, "relation2id.txt"))
            save_triples(df_triples_new, os.path.join(new_benchmark_dir, "train2id.txt"))
            save_synonyms(df_synonyms, os.path.join(new_benchmark_dir, "synonyms_uri.txt"), True)
            save_synonyms(df_synonyms, os.path.join(new_benchmark_dir, "synonyms_id.txt"), False)

            # copy entity2id.txt from source benchmark to new benchmark
            copyfile(os.path.join(benchmark_dir, "entity2id.txt"), \
                    os.path.join(new_benchmark_dir, "entity2id.txt"))

    # done
    print("DONE")

def main():
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("INPUT_BENCHMARK", type=str, \
            help="Input Benchmark to inject synonyms to.")
    parser.add_argument("-p", "--percentage-per-relation", type=float, default=0.5, \
            help="Percentage in range [0, 1] of relations to replace with one new" \
            + " artificial synonym relation (Default: 0.5).")
    parser.add_argument("-o", "--min-relation-occurence", type=int, default=1000, \
            help="Only injecting artificial synonym relations for relations which" \
            + " occur at least this often in its benchmark (Default: 1000).")
    parser.add_argument("-f", "--func_inject_synonym", type=str, \
            choices=["inject_synonym_1", "inject_synonym_2"], default="inject_synonym_1", \
            help="The injection func to use (Default: inject_synonym_1)")
    args = parser.parse_args()

    # check func
    func_inject_synonym = None
    if args.func_inject_synonym == "inject_synonym_1":
        func_inject_synonym = inject_synonym_1
    elif args.func_inject_synonym == "inject_synonym_2":
        func_inject_synonym = inject_synonym_2

    # inject synonyms to benchmark
    inject_benchmark(args.INPUT_BENCHMARK, \
            [args.percentage_per_relation], \
            [args.min_relation_occurence], \
            func_inject_synonym)

if __name__ == "__main__":
    main()

