import strlearn as sl
import numpy as np

from ensembles import DSCB
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn import under_sampling

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


def compute(clf_name, clf, chunk_size, n_chunks, experiment_name, n_drifts, stream_name):

    logging.basicConfig(filename='experiment_main.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')

    try:
        warnings.filterwarnings("ignore")

        drift = stream_name.split("/")[-2]


        filename = "results/raw_conf/%s/%s/%s/%s.csv" % (experiment_name, drift, stream_name.split("/")[-1], clf_name)
        if os.path.exists(filename):
            # print(filename)
            return

        print("START: %s, %s, %s" % (drift, stream_name, clf_name))
        logging.info("START - %s, %s, %s" % (drift, stream_name,  clf_name))
        start = time()

        stream = sl.streams.ARFFParser(stream_name, chunk_size, n_chunks)

        evaluator = sl.evaluators.TestThenTrain()
        evaluator.process(stream, clone(clf))

        stream_name = stream_name.split("/")[-1]
        filename = "results/raw_conf/%s/%s/%s/%s.csv" % (experiment_name, drift, stream_name, clf_name)
        if not os.path.exists("results/raw_conf/%s/%s/%s/" % (experiment_name, drift, stream_name)):
            os.makedirs("results/raw_conf/%s/%s/%s/" % (experiment_name, drift, stream_name))
        np.savetxt(fname=filename, fmt="%d, %d, %d, %d", X=evaluator.confusion_matrix[0])

        end = time()-start

        print("DONE: %s, %s, %s (Time: %d [s])" % (drift, stream_name, clf_name, end))
        logging.info("DONE - %s, %s, %s (Time: %d [s])" % (drift, stream_name, clf_name, end))

    except Exception as ex:
        logging.exception("Exception in %s, %s, %s" % (drift, stream_name, clf_name))
        print("ERROR: %s, %s, %s" % (drift, stream_name, clf_name))
        traceback.print_exc()
        # print(str(ex))


directory = "param_setup/"
mypath = "streams/%s" % directory
streams = ["%s%s" % (mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

chunk_size = 500
n_chunks = 20
n_drifts = 1


experiments = {
                "us-svm": SVC(probability=True),
                "us-knn": KNeighborsClassifier(),
                "us-gnb": GaussianNB(),
                "us-dtc": DecisionTreeClassifier(),
              }

experiment_names = list(experiments.keys())
base_estimators = list(experiments.values())


for base_estimator, experiment_name in zip(base_estimators, experiment_names):

    methods = {
               "DSCB-CC": DSCB(base_estimator, undersampling=under_sampling.ClusterCentroids()),
               "DSCB-CNN": DSCB(base_estimator, undersampling=under_sampling.CondensedNearestNeighbour()),
               "DSCB-ENN": DSCB(base_estimator, undersampling=under_sampling.EditedNearestNeighbours()),
               "DSCB-RENN": DSCB(base_estimator, undersampling=under_sampling.RepeatedEditedNearestNeighbours()),
               "DSCB-AKNN": DSCB(base_estimator, undersampling=under_sampling.AllKNN()),
               "DSCB-IHS": DSCB(base_estimator, undersampling=under_sampling.InstanceHardnessThreshold()),
               "DSCB-NM": DSCB(base_estimator, undersampling=under_sampling.NearMiss()),
               "DSCB-NCR": DSCB(base_estimator, undersampling=under_sampling.NeighbourhoodCleaningRule()),
               "DSCB-OSS": DSCB(base_estimator, undersampling=under_sampling.OneSidedSelection()),
               "DSCB-RUS": DSCB(base_estimator, undersampling=under_sampling.RandomUnderSampler()),
               "DSCB-TL": DSCB(base_estimator, undersampling=under_sampling.TomekLinks()),
              }

    clfs = list(methods.values())
    names = list(methods.keys())

    Parallel(n_jobs=-1)(delayed(compute)
                        (clf_name, clf, chunk_size, n_chunks, experiment_name, n_drifts, stream_name)
                        for clf_name, clf in zip(names, clfs)
                        for stream_name in streams
                        )

logging.info("-------------------")
logging.info("EXPERIMENT FINISHED")
logging.info("-------------------")
