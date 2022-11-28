import pandas as pd
from time import time
import argparse
import logging
import os
import itertools

from anonygraph.constants import *
import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
import anonygraph.time_graph_generators as generators
import anonygraph.algorithms.clustering as calgo

logging.getLogger("sklearn_extra").setLevel(logging.WARNING)
logger = logging.getLogger(__file__)


def add_arguments(parser):
    rutils.add_graph_generalization_argument(parser)
    rutils.add_log_argument(parser)

def run_generate_anonymized_clusters(args):
    return rutils.run_python_file("anonymize_clusters.py", args)

def run_generate_anonymized_kg(args):
    return rutils.run_python_file("anonymize_kg.py", args)

def run_generate_history(args):
    return rutils.run_python_file("anonymize_history.py", args)

def write_stats_to_file(path, stats):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    df = pd.DataFrame(stats)
    df.to_csv(path)

def main(args):
    logger.debug(args)
    t_min = args["min_t"]
    t_max = args["max_t"]

    # find all instances
    time_instances = sorted(utils.get_all_time_instances(args["data"], args["sample"], args["strategy"], args))


    if t_max == -1:
        # count num of
        t_max = max(time_instances)
        # logger.debug(time_instances)

    # raise Exception()
    # stats = []
    for t_index in range(len(time_instances)):
        if t_index < t_min:
            continue
        if t_index > t_max:
            break

        t = time_instances[t_index]
        current_stats = {
            "t": t
        }

        current_args = args.copy()
        current_args["t"] = t
        # run generate clusters
        start_time = time()
        result = run_generate_anonymized_clusters(current_args)
        logger.info("return code from generate clusters: {}".format(result))
        if result != 0:
            raise Exception("Errors in clustering (returned code: {})".format(result))
        current_stats["clustering"] = time() - start_time

        # run generate graph
        start_time = time()
        if args["anony_mode"] == CLUSTERS_AND_GRAPH_ANONYMIZATION_MODE:
            result = run_generate_anonymized_kg(current_args)
            if result != 0:
                raise Exception("Errors in generalizing (returned code: {})".format(result))
        current_stats["gen_kg"] = time() - start_time
        # run generate history

        start_time = time()
        result = run_generate_history(current_args)
        if result != 0:
            raise Exception("Errors in writing history (returned code: {})".format(result))

        current_stats["gen_history"] = time() - start_time

        stats_path = putils.get_performance_stats_data_path(args["data"], args["sample"], args["strategy"], args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], args["galgo"], args["anony_mode"], t, args)


        # stats.append(current_stats)

        # save stats

        write_stats_to_file(stats_path, [current_stats])

if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
