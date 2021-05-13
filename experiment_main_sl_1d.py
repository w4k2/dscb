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


def compute(clf_name, clf, drift, noise, imbalance_ratio, random_state, experiment_name, concept_kwargs):

    logging.basicConfig(filename='experiment_main.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')

    try:
        warnings.filterwarnings("ignore")

        if drift == 'incremental':
            concept_kwargs["incremental"] = True
            concept_kwargs["concept_sigmoid_spacing"] = 5

        n_drifts = concept_kwargs["n_drifts"]
        stream_size = (concept_kwargs["n_chunks"]*concept_kwargs["chunk_size"]) / 1000
        stream_name = "stream_sl_%dd_%s_%03dk_f%02d_b%02d_n%02d_rs%03d" % (n_drifts, drift[0], stream_size, concept_kwargs["n_features"], imbalance_ratio[0]*100, noise*100, random_state)

        filename = "results/raw_conf/%s/sl_%dd/%s/%s/%s.csv" % (experiment_name, n_drifts, drift, stream_name, clf_name)

        # if os.path.exists(filename):
        #     return

        print("START: %s, %s, %s, %s" % (drift, stream_name, clf_name, experiment_name))
        logging.info("START: %s, %s, %s, %s" % (drift, stream_name, clf_name, experiment_name))
        start = time()

        stream = sl.streams.StreamGenerator(**concept_kwargs, y_flip=noise, weights=imbalance_ratio, random_state=random_state)

        evaluator = sl.evaluators.TestThenTrain()
        evaluator.process(stream, clone(clf))

        filename = "results/raw_conf/%s/sl_%dd/%s/%s/%s.csv" % (experiment_name, n_drifts, drift, stream_name, clf_name)
        if not os.path.exists("results/raw_conf/%s/sl_%dd/%s/%s/" % (experiment_name, n_drifts, drift, stream_name)):
            os.makedirs("results/raw_conf/%s/sl_%dd/%s/%s/" % (experiment_name, n_drifts, drift, stream_name))
        np.savetxt(fname=filename, fmt="%d, %d, %d, %d", X=evaluator.confusion_matrix[0])

        end = time()-start

        print("DONE: %s, %s, %s, %s (Time: %d [s])" % (drift, stream_name, clf_name, experiment_name, end))
        logging.info("DONE - %s, %s, %s, %s (Time: %d [s])" % (drift, stream_name, clf_name, experiment_name, end))

    except Exception as ex:
        logging.exception("Exception in %s, %s, %s, %s" % (drift, stream_name, clf_name, experiment_name))
        print("ERROR: %s, %s, %s, %s" % (drift, stream_name, clf_name, experiment_name))
        traceback.print_exc()
        print(str(ex))


random_states = [1111, 2222, 3333, 4444, 5555]
drifts = ['sudden', 'incremental']
noises = [0.0, 0.01, 0.05]
imbalance_ratios = [
    [0.95, 0.05],
    [0.90, 0.10],
    [0.85, 0.15],
    [0.80, 0.20],
    [0.70, 0.30],
]
concept_kwargs = {
    "n_chunks": 200,
    "chunk_size": 500,
    "n_classes": 2,
    "n_drifts": 1,
    "n_features": 10,
    "n_informative": 8,
    "n_redundant": 2,
    "n_repeated": 0,
}


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

    clfs = list(methods.values())
    names = list(methods.keys())

    Parallel(n_jobs=-1)(delayed(compute)
                        (clf_name, clf, drift, noise, imbalance_ratio, random_state, experiment_name, concept_kwargs)
                        for clf_name, clf in zip(names, clfs)
                        for drift in drifts
                        for imbalance_ratio in imbalance_ratios
                        for noise in noises
                        for random_state in random_states
                        )

logging.info("-------------------")
logging.info("EXPERIMENT FINISHED")
logging.info("-------------------")
