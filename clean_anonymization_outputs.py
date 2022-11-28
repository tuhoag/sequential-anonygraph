import argparse
import logging
import os
import itertools
import glob
import shutil

from tqdm import tqdm

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
import anonygraph.time_graph_generators as generators
import anonygraph.algorithms.clustering as calgo

logger = logging.getLogger(__file__)


def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_sequence_data_argument(parser)
    parser.add_argument("--anony_mode")
    rutils.add_log_argument(parser)
    rutils.add_workers_argument(parser)

def remove_all_clusters(data_name, sample, strategy, anony_mode, args):
    clusters_path = putils.get_all_clusters_path(data_name, sample, strategy, anony_mode, args)

    file_paths = glob.glob(clusters_path + "/*.txt")
    logger.debug("removing clusters: {}".format(file_paths))

    remove_file_paths(file_paths)

def remove_all_histories(data_name, sample, strategy, anony_mode, args):
    history_dir_path = putils.get_history_tables_path(data_name, sample, strategy, anony_mode, args)

    file_paths = glob.glob(history_dir_path + "/*.json")
    logger.debug("removing histories: {}".format(file_paths))

    remove_file_paths(file_paths)

def remove_all_fake_entity_files(data_name, sample, strategy, anony_mode, args):
    fake_entities_dir_path = putils.get_fake_entities_path(data_name, sample, strategy, anony_mode, args)

    file_paths = glob.glob(fake_entities_dir_path + "/*.idx")
    logger.debug("removing fake entities files: {}".format(file_paths))

    remove_file_paths(file_paths)

def remove_all_anonymized_subgraphs(data_name, sample, strategy, anony_mode, args):
    fake_entities_dir_path = putils.get_all_anonymized_subgraphs_dir(data_name, sample, strategy, anony_mode, args)

    paths = glob.glob(fake_entities_dir_path + "/*")
    logger.debug("removing anonymized subgraphs: {}".format(paths))
    remove_folder_paths(paths)

def remove_all_stats(data_name, sample, strategy, anony_mode, args):
    stats_dir_path = putils.get_all_performance_stats_path(data_name, sample, strategy, anony_mode, args)

    paths = glob.glob(stats_dir_path + "/*.csv")
    logger.debug("removing anonymized subgraphs: {}".format(paths))
    remove_file_paths(paths)

def remove_folder_paths(paths):
    logger.info("removing {} folder paths".format(len(paths)))
    for path in tqdm(paths):
        shutil.rmtree(path)


def remove_file_paths(file_paths):
    logger.info("removing {} file paths".format(len(file_paths)))
    for path in tqdm(file_paths):
        os.remove(path)

def main(args):
    logger.info(args)
    data_name, sample, strategy, anony_mode = args["data"], args["sample"], args["strategy"], args["anony_mode"]

    remove_all_clusters(data_name, sample, strategy, anony_mode, args)
    remove_all_histories(data_name, sample, strategy, anony_mode,args)
    remove_all_fake_entity_files(data_name, sample, strategy, anony_mode, args)
    remove_all_anonymized_subgraphs(data_name, sample, strategy, anony_mode, args)
    remove_all_stats(data_name, sample, strategy, anony_mode, args)

if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
