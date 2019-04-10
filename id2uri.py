import os
import sys
from argparse import ArgumentParser

def get_rel_uris(fn_rel2id):
    f = open(fn_rel2id, "r")
    rel_uris = f.readlines()[1:]
    f.close()
    rel_uris = list(map(lambda x: x.split()[0], rel_uris))
    return rel_uris

def get_pairs_id(fn_pairs, has_header=False):
    f = open(fn_pairs, "r")
    pairs = f.readlines()[1 if has_header else 0:]
    f.close()
    pairs = list(map(lambda x: list(map(lambda y: int(y), x.split())), pairs))
    return pairs

def save_pairs_uris(fn_output, pairs_id, rel_uris):
    f = open(fn_output, "w")
    f.write(str(len(pairs_id)) + "\n")
    f.writelines(map(lambda x: rel_uris[x[0]] + "\t" + rel_uris[x[1]] + "\n", pairs_id))
    f.flush()
    f.close()

def main():
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("IDFILE", type=str, \
            help="The file with ID pairs to convert to URI pairs.")
    parser.add_argument("REL2IDFILE", type=str, \
            help="The rel2id.txt file to use as a reference to convert ID pairs.")
    parser.add_argument("-w", "--idfile-with-header", action="store_true", default=False, \
            help="An optional flag indicating wether the input ID file has a header" \
            + " in the first line containing the number of pairs (Default: False).")
    args = parser.parse_args()

    # check files
    if not os.path.exists(args.IDFILE):
        print("ERROR: specified IDFILE not found.")
        sys.exit(-1)
    if not os.path.exists(args.REL2IDFILE):
        print("ERROR: specified REL2IDFILE not found.")
        sys.exit(-1)

    # get dirname and basename
    dirname = os.path.dirname(args.IDFILE)
    basename = os.path.splitext(os.path.basename(args.IDFILE))[0]
    
    # read
    rel_uris = get_rel_uris(args.REL2IDFILE)
    pairs_id = get_pairs_id(args.IDFILE)

    # save to new uri file
    fn_output = os.path.join(dirname, basename + "_uris.txt")
    sys.stdout.write("Saving URI pairs to {0} ... ".format(fn_output))
    sys.stdout.flush()
    save_pairs_uris(fn_output, pairs_id, rel_uris)
    print("Done")

if __name__ == "__main__":
    main()

