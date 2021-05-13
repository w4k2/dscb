from core import calculate_metrics
from core import plot_streams_matplotlib
from core import pairs_metrics_multi

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

stream_sets = []
streams_aliases = []

# -------------------------------------------------------------------

streams = []
directory = "sl_1d/incremental/"
mypath = "results/raw_conf/svm/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

directory = "moa_1d/incremental/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

stream_sets += [streams]
streams_aliases += ["incremental"]
#
# # -------------------------------------------------------------------
#
streams = []
directory = "sl_1d/sudden/"
mypath = "results/raw_conf/svm/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]
#
directory = "moa_1d/sudden/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
#
stream_sets += [streams]
streams_aliases += ["sudden"]

# -------------------------------------------------------------------

streams = []
directory = "sl_1d_dyn/incremental/"
mypath = "results/raw_conf/svm/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

stream_sets += [streams]
streams_aliases += ["dynamic_drift_inc"]
#
streams = []
directory = "sl_1d_dyn/sudden/"
mypath = "results/raw_conf/svm/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

stream_sets += [streams]
streams_aliases += ["dynamic_drift_sud"]

# -------------------------------------------------------------------
#
streams = []
streams += ["real/covtypeNorm-1-2vsAll-pruned"]
streams += ["real/poker-lsn-1-2vsAll-pruned"]

stream_sets += [streams]
streams_aliases += ["real"]

# -------------------------------------------------------------------

method_names = [
                "DSCB",
                "L++CDS",
                "L++NIE",
                "KMC",
                "REA",
                "OUSE",
                "MLPC",
                ]

methods_alias = [
                "DSCB",
                "L++CDS",
                "L++NIE",
                "KMC",
                "REA",
                "OUSE",
                "MLPC",
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
                    "svm",
                    "knn",
                    "gnb",
                    "dtc"
                    ]

for streams, streams_alias in zip(stream_sets, streams_aliases):
    for experiment_name in experiment_names:
        calculate_metrics(method_names, streams, metrics, experiment_name, recount=True)
        plot_streams_matplotlib(method_names, streams, metrics, experiment_name, gauss=5, methods_alias=methods_alias, metrics_alias=metrics_alias)

    pairs_metrics_multi(method_names, streams, metrics, experiment_names, methods_alias=methods_alias, metrics_alias=metrics_alias, streams_alias=streams_alias, title=False)
