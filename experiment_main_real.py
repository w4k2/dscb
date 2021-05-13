import strlearn as sl
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.base import clone

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from ensembles import DSCB

from joblib import Parallel, delayed
from time import time

import logging
import traceback
import warnings
import os

warnings.simplefilter("ignore")


logging.basicConfig(filename='experiment_main.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')
logging.info("")
logging.info("----------------------")
logging.info("NEW EXPERIMENT STARTED")
logging.info(os.path.basename(__file__))
logging.info("----------------------")
logging.info("")


def compute(clf_name, clf, chunk_size, n_chunks, experiment_name, stream_name):

    # print(clf_name, clf, chunk_size, n_chunks, experiment_name, stream_name)

    logging.basicConfig(filename='experiment_main.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')

    try:
        warnings.filterwarnings("ignore")

        drift = stream_name.split("/")[-2]
        #
        # if os.path.exists("results/raw_conf/%s/%s/%s/%s.csv" % (experiment_name, drift, stream_name.split("/")[-1][0:-5], clf_name)):
        #     return

        print("START: %s, %s, %s, %s" % (drift, stream_name, clf_name, experiment_name))
        logging.info("START: %s, %s, %s, %s" % (drift, stream_name, clf_name, experiment_name))
        start = time()

        # print(clf.alpha, clf.beta, clf.gamma, clf.delta)

        stream = sl.streams.ARFFParser(stream_name, chunk_size, n_chunks)

        evaluator = sl.evaluators.TestThenTrain(verbose=False)
        evaluator.process(stream, clone(clf))

        stream_name = stream_name.split("/")[-1][0:-5]

        filename = "results/raw_conf/%s/%s/%s/%s.csv" % (experiment_name, drift, stream_name, clf_name)
        if not os.path.exists("results/raw_conf/%s/%s/%s/" % (experiment_name, drift, stream_name)):
            os.makedirs("results/raw_conf/%s/%s/%s/" % (experiment_name, drift, stream_name))
        np.savetxt(fname=filename, fmt="%d, %d, %d, %d", X=evaluator.confusion_matrix[0])

        end = time()-start

        print("DONE: %s, %s, %s, %s (Time: %d [s])" % (drift, stream_name, clf_name, experiment_name, end))
        logging.info("DONE - %s, %s, %s, %s (Time: %d [s])" % (drift, stream_name, clf_name, experiment_name, end))

    except Exception as ex:
        logging.exception("Exception in %s, %s, %s, %s" % (drift, stream_name, clf_name, experiment_name))
        print("ERROR: %s, %s, %s, %s" % (drift, stream_name, clf_name, experiment_name))
        traceback.print_exc()
        print(str(ex))


streams = []  # name                                                             size    n_chunks
streams.append(("streams/real/covtypeNorm-1-2vsAll-pruned.arff",                 2000,   123))
streams.append(("streams/real/poker-lsn-1-2vsAll-pruned.arff",                   2000,   169))

experiments = {
                "knn": KNeighborsClassifier(),
                "gnb": GaussianNB(),
                "dtc": DecisionTreeClassifier(),
                "svm": SVC(probability=True),
              }

experiment_names = list(experiments.keys())
base_estimators = list(experiments.values())


for base_estimator, experiment_name in zip(base_estimators, experiment_names):

    methods = {
               "DSCB": DSCB(base_estimator),
               "KMC": sl.ensembles.KMC(base_estimator=base_estimator),
               "L++CDS": sl.ensembles.LearnppCDS(base_estimator=base_estimator),
               "L++NIE": sl.ensembles.LearnppNIE(base_estimator=base_estimator),
               "REA": sl.ensembles.REA(base_estimator=base_estimator),
               "OUSE": sl.ensembles.OUSE(base_estimator=base_estimator),
               "MLPC": MLPClassifier(hidden_layer_sizes=(10)),
              }

    clf = list(methods.values())[0]
    clf_name = list(methods.keys())[0]

    Parallel(n_jobs=-1)(delayed(compute)
                        (clf_name, clf, chunk_size, n_chunks, experiment_name, stream_name)
                        for (stream_name, chunk_size, n_chunks) in streams
                        )

logging.info("-------------------")
logging.info("EXPERIMENT FINISHED")
logging.info("-------------------")
