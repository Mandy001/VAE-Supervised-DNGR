import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from libnrl.graph import *
from libnrl.classify import Classifier, read_node_label
from libnrl.dngr import DNGR
from libnrl.vaes_dngr import VAESDNGR
from libnrl.vae_dngr import VAEDNGR

from libnrl.s_dngr import SDNGR
import time
import sys

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', required=True,
                        help='Input graph file')
    parser.add_argument('--output', required=True,
                        help='Output representation file')
    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')
    parser.add_argument('--walk-length', default=80, type=int,
                        help='Length of the random walk started at each node')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--representation-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of skipgram model.')
    parser.add_argument('--epochs', default=5, type=int,
                        help='The training epochs of LINE and GCN')
    parser.add_argument('--p', default=1.0, type=float)
    parser.add_argument('--q', default=1.0, type=float)
    parser.add_argument('--method', required=True, choices=[ 'dngr','vaedngr','sdngr','vaesdngr'],
                        help='The learning method')
    # parser.add_argument('--method', required=True, choices=['grarep'],
    #                     help='The learning method')
    parser.add_argument('--label-file', default='',
                        help='The file of node label')
    parser.add_argument('--feature-file', default='',
                        help='The file of node features')
    parser.add_argument('--graph-format', default='edgelist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--negative-ratio', default=5, type=int,
                        help='the negative ratio of LINE')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--clf-ratio', default=0.5, type=float,
                        help='The ratio of training data in the classification')
    parser.add_argument('--order', default=3, type=int,
                        help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
    parser.add_argument('--no-auto-stop', action='store_true',
                        help='no early stop when training LINE')
    parser.add_argument('--dropout', default=0.5, type=float, 
                        help='Dropout rate (1 - keep probability)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix')
    parser.add_argument('--hidden', default=16, type=int,
                        help='Number of units in hidden layer 1')
    parser.add_argument('--kstep', default=4, type=int,
                        help='Use k-step transition probability matrix')
    parser.add_argument('--lamb', default=0.2, type=float,
                        help='lambda is a hyperparameter in TADW')
    args = parser.parse_args()
    return args


def main(args):
    t1 = time.time()
    g = Graph()
    print("Reading...")

    X, Y = read_node_label(args.label_file)
    training_size = int(args.clf_ratio * len(X))
    shuffle_indices = np.random.permutation(np.arange(len(X)))
    X_train = [X[shuffle_indices[i]] for i in range(training_size)]
    Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
    X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
    Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]


    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input, weighted=args.weighted, directed=args.directed)

    if args.method == 'dngr':
        model = DNGR(graph=g, Kstep=args.kstep, dim=args.representation_size, XY=[X_train, Y_train])
        model.show()

    if args.method == 'vaedngr':
        model = VAEDNGR(graph=g, Kstep=args.kstep, dim=args.representation_size, XY=[X_train, Y_train])
        model.show()

    if args.method == 'sdngr':
        model= SDNGR(graph=g, Kstep=args.kstep, dim=args.representation_size, XY=[X_train, Y_train])

    if args.method == 'vaesdngr':
        model = VAESDNGR(graph=g, Kstep=args.kstep, dim=args.representation_size, XY=[X_train, Y_train])
        model.show()


    t2 = time.time()
    print(t2-t1)
    if args.method != 'gcn':
        print("Saving embeddings...")
        model.save_embeddings(args.method+'_'+args.output)
    # if args.label_file and args.method != 'gcn':
    vectors = model.vectors

    print("Training classifier using {:.2f}% nodes...".format(args.clf_ratio*100))
    clf = Classifier(vectors=vectors, clf=LogisticRegression())
    clf.my_evaluate(X_train, Y_train, X_test, Y_test)



if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)

    temp = sys.argv

    print(temp)

    # 1 dngr          'macro': 0.51985020856984931   'micro': 0.66417290108063176,
    command = "--method dngr --label-file ../data/wiki/Wiki_category.txt --input ../data/wiki/Wiki_edgelist.txt --output wiki_vec_all.txt"

    # 2 supervised  +  dngr  'macro': 0.52338106860943268  , 'micro': 0.66500415627597675}
    #command = "--method sdngr --label-file ../data/wiki/Wiki_category.txt --input ../data/wiki/Wiki_edgelist.txt --output wiki_vec_all.txt"

    # 3 VAE +  dngr     'macro': 0.54889058783948963  , 'micro': 0.6874480465502909 }
    # command = "--method vaedngr --label-file ../data/wiki/Wiki_category.txt --input ../data/wiki/Wiki_edgelist.txt --output wiki_vec_all.txt"

    # 4 VAE + supervised + dngr    'macro': 0.5418012273597721, 'micro': 0.69326683291770574}
    #command = "--method vaesdngr --label-file ../data/wiki/Wiki_category.txt --input ../data/wiki/Wiki_edgelist.txt --output vaesdngr_wiki_vec_all.txt"


    sys.argv = sys.argv + command.split()

    main(parse_args())
