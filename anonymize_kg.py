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
import anonygraph.algorithms.generalization as gen
import anonygraph.algorithms as algo

logger = logging.getLogger(__file__)


def add_arguments(parser):
    rutils.add_graph_generalization_argument(parser)
    rutils.add_log_argument(parser)


def main(args):
    logger.info(args)
    # load clusters
    # clusters_path = putils.get_clusters_path(
    #     args["data"], args["sample"], args["strategy"], args["t"],
    #     args["info_loss"], args["k"], args["w"], args["calgo"], args["anony_mode"], args
    # )
    clusters_path = putils.get_clusters_path(
        args["data"], args["sample"], args["strategy"], args["t"],
        args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], args["anony_mode"], args
    )
    clusters = algo.Clusters.from_file(clusters_path)

    logger.info(
        "[time: {}] loaded clusters from: {}".format(args["t"], clusters_path)
    )
    logger.debug("loaded clusters: {}".format(clusters))

    # load original subgraph
    subgraph = dutils.load_raw_subgraph(
        args["data"], args["sample"], args["strategy"], args["t"], args
    )
    logger.info("[time: {}] loaded subgraph: {}".format(args["t"], subgraph))

    # load current fake entity manager
    # fake_entity_manager = utils.get_fake_entity_manager(
    #     args["data"], args["sample"], args["strategy"], args["t"],
    #     args["info_loss"], args["k"], args["w"], args["calgo"], args["galgo"], args["anony_mode"],
    #     args
    # )
    fake_entity_manager = utils.get_fake_entity_manager(args["data"], args["sample"], args["strategy"], args["t"], args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], None, args["anony_mode"], args)

    # load algo to generate graphs
    gen_fn = gen.GraphGeneralization(args["galgo"], fake_entity_manager)
    anony_subgraph = gen_fn.run(subgraph, clusters)
    logger.debug("final anonymized subgraph: {}".format(anony_subgraph))

    # save anonymized subgraph
    anony_subgraph_path = putils.get_anonymized_subgraph_path(
        args["data"], args["sample"], args["strategy"], args["t"],
        args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], args["galgo"], args["anony_mode"],
        args
    )

    anony_subgraph.to_edge_files(anony_subgraph_path)
    logger.info(
        "[time: {}] saved anonymized subgraph to {}".format(
            args["t"], anony_subgraph_path
        )
    )

    # fake_entity_path = putils.get_fake_entity_path(
    #     args["data"], args["sample"], args["strategy"], args["t"],
    #     args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], args["galgo"], args["anony_mode"],
    #     args
    # )
    # fake_entity_manager.to_file(fake_entity_path)
    # logger.info(
    #     "[time: {}] updated fake entities at: {}".format(
    #         args["t"], fake_entity_path
    #     )
    # )

    sys.exit(0)


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
