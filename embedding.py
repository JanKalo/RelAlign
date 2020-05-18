#!/usr/bin/python -u

# force matplotlib agg backend
import matplotlib
matplotlib.use("agg")

from thirdParty.OpenKE import config
from thirdParty.OpenKE import models
import numpy as np
import pandas as pd
import sys
import os

class Embedding(object):
    MODELS = {"rescal": models.RESCAL, \
            "transe": models.TransE, \
            "transh": models.TransH, \
            "transr": models.TransR, \
            "transd": models.TransD, \
            "distmult": models.DistMult, \
            "hole": models.HolE, \
            "complex": models.ComplEx, \
            "analogy": models.Analogy}

    def __init__(self, benchmark_dir, embedding_dir, \
            embedding_model, embedding_dimensions=100, \
            work_threads=32, include_test_triples=False):
        self.benchmark_dir = benchmark_dir.rstrip("/")
        self.embedding_dir = embedding_dir.rstrip("/")
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.work_threads = work_threads
        self.include_test_triples = include_test_triples
        self.con = None
        self.__init()
        self.__init_benchmark()

    def __init(self):
        print("### IMPORTING EMBEDDING")
        print("### Benchmark    : %s" % self.benchmark_dir)
        print("### Embedding    : %s" % self.embedding_dir)
        print("### Dimensions   : %d" % self.embedding_dimensions)
        print("### Work Threads : %d" % self.work_threads)
        self.con = config.Config()
        self.con.set_in_path(self.benchmark_dir + "/")
        self.con.set_import_files(os.path.join(self.embedding_dir, "model.vec.tf"))
        self.con.set_test_flag(self.include_test_triples)
        self.con.set_work_threads(self.work_threads)
        self.con.set_dimension(self.embedding_dimensions)
        self.con.init()
        self.con.set_model(self.embedding_model)
        self.con.import_variables()
        print("### IMPORTING EMBEDDING DONE")

    def __init_benchmark(self):
        sys.stdout.write("Reading Benchmark as source to lookup names ... ")
        sys.stdout.flush()
        entities_file = open(os.path.join(self.benchmark_dir, "entity2id.txt"), "r")
        self.entities = entities_file.readlines()[1:]
        entities_file.close()
        self.ent_ids = {}
        for line in self.entities:
            entity = line.split("\t")[0]
            ent_id = line.split("\t")[1].replace("\n", "")
            self.ent_ids[entity] = ent_id

        relations_file = open(os.path.join(self.benchmark_dir, "relation2id.txt"), "r")
        self.relations = relations_file.readlines()[1:]
        relations_file.close()
        self.rel_ids = {}
        for line in self.relations:
                relation = line.split("\t")[0]
                rel_id = line.split("\t")[1].replace("\n", "")
                self.rel_ids[relation] = rel_id
        print("Done")

    def lookup_ent_id(self, entity):
        if entity in self.ent_ids:
	        return self.ent_ids[entity]
        else:
            return None

    def lookup_entity(self, ent_id):
        return self.entities[ent_id].split('\t')[0]

    def lookup_rel_id(self, relation):
        if relation in self.rel_ids:
	        return self.rel_ids[relation]
        else:
            return None

    def lookup_relation(self, rel_id):
        return self.relations[rel_id].split('\t')[0]

    def get_df_triples(self):
        sys.stdout.write("Getting Triples from Toolkit as Pandas Dataframe ... ")
        sys.stdout.flush()
        num_triples = self.con.lib.getTripleTotal()
        subjects = np.zeros(num_triples, dtype = np.int64)
        objects = np.zeros(num_triples, dtype = np.int64)
        relations = np.zeros(num_triples, dtype = np.int64)
        self.con.lib.getTripleList(subjects.__array_interface__["data"][0],
                objects.__array_interface__["data"][0],
                relations.__array_interface__["data"][0])
        df_triples = pd.DataFrame()
        df_triples["Subject"] = pd.Series(subjects, dtype = np.int64)
        df_triples["Object"] = pd.Series(objects, dtype = np.int64)
        df_triples["Relation"] = pd.Series(relations, dtype = np.int64)
        print("Done")
        return df_triples

    def add_predictions(self, df_triples):
        sys.stdout.write("Adding Embedding Predictions to the triples in the Dataframe ... ")
        sys.stdout.flush()
        predictions = self.con.test_step(df_triples["Subject"], df_triples["Object"], df_triples["Relation"])
        df_triples["Embedding_Predict"] = pd.Series(predictions, dtype = np.float32)
        print("Done")
        return df_triples

    def get_embedding_list(self):
        return list(self.con.get_parameter_lists().keys())

    def get_df_embedding(self, embedding_key):
        sys.stdout.write("Creating Pandas Dataframe from Embedding ... ")
        sys.stdout.flush()
        embedding = self.con.get_parameters_by_name(embedding_key)
        df_embedding = pd.DataFrame(embedding, dtype = np.float32)
        print("Done")
        return df_embedding

    def get_predict(self, heads, tails, relations):
        return self.con.test_step(heads, tails, relations)

