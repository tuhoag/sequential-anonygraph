import sys
import argparse
import logging
import os
import itertools

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
    # rutils.add_clusters_generation_argument(parser)
    rutils.add_log_argument(parser)

    # parser.add_argument("--reset_w", type=int, default=-1)

def main(args):
    logger.info(args)
    # raise Exception()
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
    # raise Exception()

    logger.info("[time: {}] loading pairs distances at time: {}".format(t, t))

    # sys.exit(0)
    pair_dist = utils.get_pair_distance_of_subgraph(args["data"], args["sample"], args["strategy"], t, args["info_loss"], args)
    logger.info("[time: {}] loaded pair distance with {} pairs".format(t, len(pair_dist)))

    # read table history
    logger.info("[time: {}] loading history at time: {}".format(t, previous_t))

    history = utils.get_history_table(args["data"], args["sample"], args["strategy"], previous_t, args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], args["galgo"], args["anony_mode"], args)
    logger.info("loaded history at time {}".format(previous_t))

    fake_entity_manager = utils.get_fake_entity_manager(args["data"], args["sample"], args["strategy"], previous_t, args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], args["galgo"], args["anony_mode"], args)

    # run anonymization at time t
    logger.info("[time: {}] generating clusters".format(t))

    entity2sensitive_vals = dutils.load_sensitive_vals(args["data"], args["sample"], args["strategy"], t, args)[1]
    logger.info("loaded entity2svals at time {}".format(t))

    all_sval_ids = dutils.load_all_sensitive_vals(args["data"], args["sample"], args["sattr"], args)
    logger.info("loaded all sval ids: {}".format(all_sval_ids))

    algo = calgo.ClustersGeneration(args["k"], args["w"], args["l"],  args["calgo"], args["enforcer"], args)
    clusters = algo.run(pair_dist, entity2sensitive_vals, all_sval_ids, history, fake_entity_manager, t)

    clusters_path = putils.get_clusters_path(args["data"], args["sample"], args["strategy"], t, args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], args["anony_mode"], args)
    clusters.to_file(clusters_path)
    logger.info("[time: {}] saved clusters to: {}".format(t, clusters_path))
    logger.debug(clusters)

    fake_entity_path = putils.get_fake_entity_path(args["data"], args["sample"], args["strategy"], t, args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], args["galgo"], args["anony_mode"], args)
    logger.info("[time: {}] saving fake entities at: {}".format(t, fake_entity_path))
    fake_entity_manager.to_file(fake_entity_path)
    logger.info("[time: {}] saved fake entities at: {}".format(t, fake_entity_path))

    sys.exit(0)

if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
