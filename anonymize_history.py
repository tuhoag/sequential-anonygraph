import sys
import argparse
import logging
import os
import itertools

import anonygraph.constants as constants
import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
import anonygraph.time_graph_generators as generators
import anonygraph.algorithms.clustering as calgo
import anonygraph.algorithms as algo

logging.getLogger("sklearn_extra").setLevel(logging.WARNING)
logger = logging.getLogger(__file__)


def add_arguments(parser):
    rutils.add_graph_generalization_argument(parser)
    rutils.add_log_argument(parser)

def main(args):
    logger.info(args)

    # find all instances
    time_instances = sorted(utils.get_all_time_instances(args["data"], args["sample"], args["strategy"], args))

    t = args["t"]
    t_index = time_instances.index(t)

    reset_window = args["reset_w"]
    if reset_window == -1:
        reset_window = len(time_instances)


    logger.debug(time_instances)
    logger.debug(t_index % reset_window)
    if t_index % reset_window != 0:
        previous_t = time_instances[t_index - 1]
    else:
        previous_t = -1

    logger.debug(previous_t)

    # read table history
    logger.info("[time: {}] loading history at time: {}".format(t, previous_t))

    history = utils.get_history_table(args["data"], args["sample"], args["strategy"], previous_t, args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], args["galgo"], args["anony_mode"], args)
    # logger.info("[time: {}] loaded history at time {}".format(previous_t))

    fake_entity_manager = utils.get_fake_entity_manager(args["data"], args["sample"], args["strategy"], args["t"], args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], args["galgo"], args["anony_mode"], args)
    logger.debug(fake_entity_manager)
    # raise Exception()

    if args["anony_mode"] == constants.CLUSTERS_AND_GRAPH_ANONYMIZATION_MODE:
        # logger.debug("generating history table from subgraph")
        # anony_subgraph = dutils.get_anonymized_subgraph(args["data"], args["sample"], args["strategy"], args["t"], args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], args["galgo"], args["anony_mode"], args)
        # logger.info("[time: {}] loaded anonymized subgraph".format(t))

        # history.add_new_entry_from_subgraph(anony_subgraph, t)
        logger.debug("generating history table from clusters")
        clusters_path = putils.get_clusters_path(
            args["data"], args["sample"], args["strategy"], args["t"],
            args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], args["anony_mode"], args
        )
        clusters = algo.Clusters.from_file(clusters_path)
        logger.debug("loaded clusters at {}".format(clusters_path))
        raw_subgraph = dutils.load_raw_subgraph(
            args["data"], args["sample"], args["strategy"], args["t"], args
        )
        logger.debug("loaded raw subgraph at time: {}".format(args["t"]))

        history.add_new_entry_from_clusters(clusters, raw_subgraph, fake_entity_manager, t)

    elif args["anony_mode"] == constants.CLUSTERS_ANONYMIZATION_MODE:
        logger.debug("generating history table from clusters")
        clusters_path = putils.get_clusters_path(
            args["data"], args["sample"], args["strategy"], args["t"],
            args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], args["anony_mode"], args
        )
        clusters = algo.Clusters.from_file(clusters_path)
        logger.debug("loaded clusters at {}".format(clusters_path))
        raw_subgraph = dutils.load_raw_subgraph(
            args["data"], args["sample"], args["strategy"], args["t"], args
        )
        logger.debug("loaded raw subgraph at time: {}".format(args["t"]))

        history.add_new_entry_from_clusters(clusters, raw_subgraph, fake_entity_manager, t)

    # save information to history table file
    history_path = putils.get_history_table_path(args["data"], args["sample"], args["strategy"], t, args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], args["galgo"], args["anony_mode"], args)
    logger.info("[time: {}] saving history at: {}".format(t, history_path))
    history.to_file(history_path)

    sys.exit(0)

if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
