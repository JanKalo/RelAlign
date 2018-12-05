from thirdParty.OpenKE import config
from thirdParty.OpenKE import models
from embedding import Embedding
import time
import sys
import os
from argparse import ArgumentParser

def train(benchmark_folder, embedding_folder, train_model_str, test_flag=False, \
        work_threads=32, epoch_count=1000, batch_count=100, learning_rate=0.0001, \
        margin=1.0, bern=0, dimension=100, ent_negative_rate=4, rel_negative_rate=0, \
        optimization_method="SGD"):
    # output (hyper-) parameters
    print("###### (HYPER-) PARAMETERS: ######")
    print("Working Threads:\t%d" % (work_threads))
    print("Epoch Count:\t\t%d" % (epoch_count))
    print("Batch Count Per Epoch:\t%d" % (batch_count))
    print("Learning Rate:\t\t%f" % (learning_rate))
    print("Margin:\t\t\t%f" % (margin))
    print("Bern:\t\t\t%d" % (bern))
    print("Embedding Dimensions:\t%d" % (dimension))
    print("Negative Entity Rate:\t%d" % (ent_negative_rate))
    print("Negative Relation Rate:\t%d" % (rel_negative_rate))
    print("Optimization Method:\t%s" % (optimization_method))
    print("")
    print("############# SETUP ##############")

    # get train model
    train_model = Embedding.MODELS[train_model_str]

    # set benchmark
    con = config.Config()
    benchmark_folder = benchmark_folder.rstrip("/")
    con.set_in_path(benchmark_folder + "/")
    con.set_test_flag(test_flag)
    
    # set (hyper-) parameters
    con.set_work_threads(work_threads)
    con.set_train_times(epoch_count)
    con.set_nbatches(batch_count)
    con.set_alpha(learning_rate)
    con.set_margin(margin)
    con.set_bern(bern)
    con.set_dimension(dimension)
    con.set_ent_neg_rate(ent_negative_rate)
    con.set_rel_neg_rate(rel_negative_rate)
    con.set_opt_method(optimization_method)

    # prepare embedding folder
    embedding_folder = embedding_folder.rstrip("/")

    # models will be exported via tf.Saver() automatically
    model_filename = os.path.join(embedding_folder, "model.vec.tf")
    con.set_export_files(model_filename, 10)
    
    # model parameters will be exported to json files automatically
    # (disabled because this will redundantly save parameters as json in a much larger file)
    #embedding_filename = os.path.join(embedding_dir, "embedding.vec.json")
    #con.set_out_files(embedding_filename)
    
    # initialize experimental settings
    con.init()
    
    # set the knowledge embedding model
    con.set_model(train_model)
    
    # import variables from model if existing (to continue training)
    if os.path.exists(embedding_folder):
        con.set_import_files(model_filename)
        con.import_variables()
    else:
        os.mkdir(embedding_folder)
    
    # train the model and record time
    print("")
    print("############# TRAIN ##############")
    print("### TRAIN MODEL: %s" % train_model_str)
    start = time.time()
    print("### START TIME: %s" % (time.ctime(start)))
    con.run()
    end = time.time()
    print("")
    print("### FINISHED. END TIME: %s" % (time.ctime(end)))
    print("### ELAPSED TIME: %f seconds" % (end - start))
    
    # test the model if test flag set
    if test_flag:
        print("")
        print("############# TEST ###############")
        con.test()

OPTIMIZER = ["SGD", "Adagrad", "Adadelta", "Adam"]

def main():
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("MODEL_TYPE", type=str, choices=Embedding.MODELS.keys(), \
            help="The model type to train the benchmark with")
    parser.add_argument("INPUT_BENCHMARK", type=str, \
            help="The benchmark directory containing the benchmark with the" \
            + " 'entity2id.txt', 'relation2id.txt', 'train2id.txt'" \
            + " ('valid2id.txt', 'test2id.txt') files to train")
    parser.add_argument("OUTPUT_EMBEDDING", type=str, \
            help="The output directory where to save the embedding")
    parser.add_argument("-d", "--dimension", type=int, default=100, \
            help="The number of hidden dimensions in the output embedding (Default: 100)")
    parser.add_argument("-c", "--epoch-count", type=int, default=1000, \
            help="The number of epochs to train (Default: 1000)")
    parser.add_argument("-f", "--batch-count", type=int, default=100, \
            help="The number of batches per epoch (NOT the size of batches) (Default: 100)")
    parser.add_argument("-l", "--learning-rate", type=float, default=0.001, \
            help="The learning rate (Default: 0.001)")
    parser.add_argument("-o", "--optimizer", type=str, choices=OPTIMIZER, \
            default="SGD", \
            help="The optimization method to use (Default: SGD)")
    parser.add_argument("-b", "--bern", action="store_true", default=False, \
            help="Use bern for negative sampling (Default: False)")
    parser.add_argument("-e", "--neg-entity-rate", type=int, default=4, \
            help="The rate of negative entities per triple (Default: 4)")
    parser.add_argument("-r", "--neg-relation-rate", type=int, default=0, \
            help="The rate of negative relations per triple (Default: 0)")
    parser.add_argument("-m", "--margin", type=float, default=1.0, \
            help="The margin between positive and negative instances (Default: 1.0)")
    parser.add_argument("-t", "--test-flag", action="store_true", default=False, \
            help="Test flag for testing after training." \
            + " This will need 'valid2id.txt' and 'test2id.txt' (Default: False)")
    parser.add_argument("-w", "--work-threads", type=int, default=32, \
            help="The number of work threads to use for training (Default: 32)")
    args = parser.parse_args()
    
    # check benchmark folder
    if not os.path.exists(args.INPUT_BENCHMARK):
        sys.exit("Benchmark does not exist, exiting")
    
    # train
    train(benchmark_folder=args.INPUT_BENCHMARK, \
            embedding_folder=args.OUTPUT_EMBEDDING, \
            train_model_str=args.MODEL_TYPE, \
            test_flag=args.test_flag, \
            work_threads=args.work_threads, \
            epoch_count=args.epoch_count, \
            batch_count=args.batch_count, \
            learning_rate=args.learning_rate, \
            margin=args.margin, \
            bern=int(args.bern), \
            dimension=args.dimension, \
            ent_negative_rate=args.neg_entity_rate, \
            rel_negative_rate=args.neg_relation_rate, \
            optimization_method=args.optimizer)

if __name__ == "__main__":
    main()

