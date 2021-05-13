from core import calculate_metrics
from core import plot_table_matplotlib_params

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

stream_sets = []
streams_aliases = []
streams = []


directory = "param_setup/"
mypath = "streams/%s" % directory
streams += ["%s%s.arff" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
streams.sort()

stream_sets += [streams]
streams_aliases += ["sl"]


methods = {
            "DSCB-SMOTE":     "SMOTE",
            "DSCB-ADASYN":    "ADASYN",
            "DSCB-BSMOTE":    "BSMOTE",
            "DSCB-SVMSMOTE":  "SVMSMOTE",
            "DSCB-RUS":       "ROS",
}
experiment_names = {
                "os-svm": "svm",
                "os-knn": "knn",
                "os-gnb": "gnb",
                "os-dtc": "dtc",
}

# methods = {
#             "DSCB-CNN":  "CNN",
#             "DSCB-ENN":  "ENN",
#             "DSCB-RENN": "RENN",
#             "DSCB-AKNN": "AKNN",
#             "DSCB-RUS":  "RUS",
#             "DSCB-NCR":  "NCR",
#             "DSCB-OSS":  "OSS",
#             "DSCB-TL":   "TL",
#             "DSCB-NM":   "NM",
#             "DSCB-IHS":  "IHS",
# }
# experiment_names = {
#                 "us-svm": "svm",
#                 "us-knn": "knn",
#                 "us-gnb": "gnb",
#                 "us-dtc": "dtc",
# }

# methods = {
#                "DSCB-0.1": "xi = 0.1",
#                "DSCB-0.2": "xi = 0.2",
#                "DSCB-0.3": "xi = 0.3",
#                "DSCB-0.4": "xi = 0.4",
#                "DSCB-0.5": "xi = 0.5",
#                "DSCB-0.6": "xi = 0.6",
#                "DSCB-0.7": "xi = 0.7",
#                "DSCB-0.8": "xi = 0.8",
#                "DSCB-0.9": "xi = 0.9",
#                "DSCB-1.0": "xi = 1.0",
# }
# experiment_names = {
#               "xi-svm": "svm",
#               "xi-knn": "knn",
#               "xi-gnb": "gnb",
#               "xi-dtc": "dtc",
# }


# methods = {
#                "DSCB-BP_25": "BP = 0.25",
#                "DSCB-BP_30": "BP = 0.30",
#                "DSCB-BP_35": "BP = 0.35",
#                "DSCB-BP_40": "BP = 0.40",
#                "DSCB-BP_45": "BP = 0.45",
#                "DSCB-BP_50": "BP = 0.50",
#  }
#
# experiment_names = {
#                  "bp-svm": "svm",
#                  "bp-knn": "knn",
#                  "bp-gnb": "gnb",
#                  "bp-dtc": "dtc",
#  }


method_names = list(methods.keys())
methods_alias = list(methods.values())


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

for streams, streams_alias in zip(stream_sets, streams_aliases):
    for experiment_name in experiment_names:
        calculate_metrics(method_names, streams, metrics, experiment_name, recount=True)

    plot_table_matplotlib_params(method_names, streams, metrics, experiment_names, metrics_alias=metrics_alias, methods_alias=methods_alias)
