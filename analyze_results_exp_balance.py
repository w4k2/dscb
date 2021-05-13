from core import calculate_metrics
from core import plot_streams_bexp

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
streams = []

directory = "balance_exp/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
streams.sort()

method_names = [
                "DSCB",
                "KMeanClustering",
                "LearnppCDS",
                "LearnppNIE",
                "REA",
                "OUSE",
                ]

methods_alias = [
                "DSCB",
                "KMC",
                "L++CDS",
                "L++NIE",
                "REA",
                "OUSE",
                ]

metrics_alias = [
           "Gmean",
           "F-score",
           "Precision",
           "Recall",
           "Specificity",
          ]

metrics = [
           "g_mean",
           "f1_score",
           "precision",
           "recall",
           "specificity",
          ]

experiment_names = [
                    "balance_exp/svm",
                    "balance_exp/knn",
                    "balance_exp/gnb",
                    "balance_exp/dtc"
                   ]

for experiment_name in experiment_names:

    calculate_metrics(method_names, streams, metrics, experiment_name, recount=True)
    plot_streams_bexp(method_names, streams, metrics, experiment_name, methods_alias=methods_alias, metrics_alias=metrics_alias)
