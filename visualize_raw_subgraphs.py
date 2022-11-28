import numpy as np
import math
from time import time
from tqdm import tqdm
from joblib import Parallel, delayed
import glob
import argparse
import logging
import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from anonygraph import data

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
import anonygraph.utils.visualization as visual
import anonygraph.time_graph_generators as generators
import anonygraph.algorithms.clustering as calgo
import anonygraph.algorithms as algo
import anonygraph.constants as constants
import anonygraph.evaluation.raw_subgraphs_metrics as rgmetrics

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def add_arguments(parser):
    # rutils.add_data_argument(parser)
    rutils.add_sequence_data_argument(parser)
    rutils.add_workers_argument(parser)
    rutils.add_log_argument(parser)

    parser.add_argument("--refresh", type=rutils.str2bool)


def get_raw_snapshot_quality(subgraph, pre_subgraph, args):
    metrics_names = rgmetrics.get_all_metric_names()

    return rgmetrics.calculate_quality_metrics(
        metrics_names, subgraph, pre_subgraph, args
    )


def get_raw_snapshot_quality_from_path(graph_path):
    info = putils.extract_info_from_raw_subgraph_path(graph_path)
    logger.debug("subgraph_path: {} - info: {}".format(graph_path, info))
    # raise Exception()

    data_name = info["data"]
    sample = info["sample"]
    strategy_name = info["strategy"]
    t = info["t"]
    pre_t = max(t - 1, 0)

    raw_subgraph = dutils.load_raw_subgraph(
        data_name, sample, strategy_name, t, info
    )

    pre_raw_subgraph = dutils.load_raw_subgraph(
        data_name, sample, strategy_name, pre_t, info
    )

    quality_info = get_raw_snapshot_quality(
        raw_subgraph, pre_raw_subgraph, info
    )
    info.update(quality_info)

    return info


def prepare_data(data_info, num_workers, args):
    data_name, sample = data_info["data"], data_info["sample"]

    dir_path = putils.get_output_path(data_name, sample)
    logger.debug("sequence of snapshots path: {}".format(dir_path))

    snapshot_paths = glob.glob(dir_path + "/*/raw/[0-9]*")
    logger.info("preparing data from {} snapshots".format(len(snapshot_paths)))
    logger.debug(snapshot_paths)
    # raise Exception()
    start_time = time()
    raw_data = list(
        Parallel(n_jobs=num_workers)(
            delayed(get_raw_snapshot_quality_from_path)(path)
            for path in tqdm(snapshot_paths)
        )
    )
    logger.debug(raw_data)
    logger.info(
        "finished preparing data in {} seconds".format(time() - start_time)
    )

    # start_time = time()
    # parts = utils.split_data_to_parts(clusters_paths, num_workers)
    # raw_data = list(itertools.chain.from_iterable(
    #     Parallel(n_jobs=num_workers)(
    #         delayed(get_clusters_quality)(part) for part in tqdm(parts)
    #     )
    # ))
    # logger.debug(raw_data)
    # logger.info("finished preparing data in {} seconds".format(time() - start_time))

    return raw_data


def add_more_info(df):
    df["strategy_key"] = df["strategy"] + "#" + df["n_sg"].map(str)
    df["t"] = df["t"] + 1

    df["data"].replace("yago15", "yago", inplace=True)

    df["ratio_changed_edges"] = (df["num_new_edges"] + df["num_removed_edges"]) / (df["num_edges"])

    df["ratio_changed_entities"] = (df["num_new_entities"] + df["num_removed_entities"]) / df["num_entities"]


def visualize(df, strategy_key):
    logger.debug(df)
    df = df[df["strategy_key"] == strategy_key]
    df.sort_values(["t"], inplace=True)
    logger.info("visualizing data: {}".format(df))

    strategy_keys = df["strategy_key"].unique()
    t_values = df["t"].unique()

    logger.debug("strategy keys: {}".format(strategy_keys))

    current_palette = sns.color_palette(n_colors=len(strategy_keys))
    plt.plot(
        "t",
        constants.RAW_ENTITIES_METRIC,
        data=df,
        # marker="o",
        linestyle="dashed",
        label="num of entities",
        color="red",
    )
    # plt.plot(
    #     "t",
    #     constants.RAW_EDGES_METRIC,
    #     data=df,
    #     # marker="^",
    #     # linestyle="",
    #     label="num of edges",
    #     color="blue",
    # )
    plt.ylabel("Data Size")
    plt.xlabel("t")
    plt.grid(linestyle="--")
    plt.xticks(t_values)

    plt.legend()
    plt.show()


def get_title(name):
    name2title = {
        "num_entities": "Number of Entities",
        "num_edges": "Number of Edges",
        "num_new_entities": "Number of New Entities",
        "num_old_entities": "Number of Old Entities",
        "num_removed_entities": "Number of Removed Entities",
        "num_new_edges": "Number of New Edges",
        "data": "data",
        "ratio_changed_edges": "Ratio of Added/Removed Edges (%)",
        "ratio_changed_entities": "Ratio of Added/Removed Nodes (%)"
    }

    return name2title[name]


def visualize_strategies(df, strategy_keys, y_name, path=None):
    df = df[df["strategy_key"].isin(strategy_keys)]
    logger.debug(df)
    df.sort_values(["t"], inplace=True)
    logger.info("visualizing data: {}".format(df))

    strategy_keys = df["strategy_key"].unique()
    t_values = df["t"].unique()

    logger.debug("strategy keys: {}".format(strategy_keys))

    current_palette = sns.color_palette(n_colors=len(strategy_keys))
    sns.lineplot(data=df, x="t", y=y_name, hue="strategy_key")

    plt.ylabel(get_title(y_name))
    plt.xlabel("t")
    plt.grid(linestyle="--")
    plt.xticks(t_values)

    plt.legend()

    if path is not None:
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(path)

    plt.show()

def get_xticks(name, values, num_sticks=-1):
    if name == "t":
        if num_sticks == -1:
            x_ticks = values
        else:
            sorted_values = sorted(values)
            step_value = math.floor(len(values) / num_sticks)
            logger.debug("step_value: {}".format(step_value))
            x_ticks = []
            for val_idx in range(0, len(sorted_values), step_value):
                # logger.debug("val_idx: {}".format(val_idx))
                x_ticks.append(sorted_values[val_idx])

            # x_ticks.append(sorted_values[-1])
            logger.debug("xticks: {}".format(x_ticks))
    else:
        raise Exception("Unsupported get_xticks of {}".format(name))

    return x_ticks

def visualize_metrics_of_all_datasets(df, x_name, y_name, cat_name, path, strategy_key):
    logger.debug(df[[x_name, cat_name, y_name]])

    df = df[(df["strategy_key"] == strategy_key)]
    cat_values = df[cat_name].unique()
    x_values = df[x_name].unique()
    num_cat_values = len(cat_values)

    logger.debug("filtered df (len: {}): {}".format(len(df), df[[x_name, cat_name, y_name]]))
    current_palette = sns.color_palette(
        n_colors=num_cat_values, palette="bright"
    )

    logger.info("y_name: {}".format(y_name))

    for cat_value in cat_values:
        current_df = df[(df[cat_name] == cat_value)]
        logger.info("cat: {} - {} values - avg: {}".format(cat_value, len(current_df), np.mean(current_df[y_name])))


    sns.lineplot(
        data=df,
        x=x_name,
        y=y_name,
        hue=cat_name,
        style=cat_name,
        palette=current_palette,
        legend=False,
    )
    plt.ylabel(get_title(y_name))
    plt.grid(linestyle="--")
    plt.xticks(get_xticks(x_name, x_values, 8))
    plt.legend(title=get_title(cat_name), labels=cat_values)
    if path is not None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        plt.savefig(path)

    plt.show()
    plt.clf()


def main(args):
    logger.debug(args)
    data_names = ["email-temp", "yago15"]
    # for each dataset, collect all its subgraphs' data
    # merge all of them

    df_list = []
    for data_name in data_names:
        data_path = putils.get_raw_snapshots_exp_data_path(data_name, -1, args)

        current_df = visual.get_exp_data(
            exp_path=data_path,
            prepare_data_fn=prepare_data,
            prepare_data_args={
                "data": data_name,
                "sample": -1,
            },
            workers=args["workers"],
            refresh=args["refresh"],
            args=args
        )

        logger.debug(
            "data: {} - max_t: {}".format(data_name, max(current_df["t"]))
        )

        df_list.append(current_df)

    df = pd.concat(df_list)

    add_more_info(df)

    logger.debug(df)
    logger.info("visualizing")
    # visualize(df, strategy_key="equalraw~50.0")
    # metric_names = rgmetrics.get_all_metric_names()
    metric_names = ["ratio_changed_edges", "ratio_changed_entities"]
    for metric_name in metric_names:
        dir_name = os.path.join(os.path.dirname(data_path), "figures")
        file_name = "raw-graphs_{}.pdf".format(metric_name)
        current_path = os.path.join(dir_name, file_name)

        # logger.info("metric: {} - avg: {}".format(metric_name, np.mean(df[metric_name])))

        visualize_metrics_of_all_datasets(
            df=df,
            x_name="t",
            y_name=metric_name,
            cat_name="data",
            path=current_path,
            strategy_key = "mean#20",
        )
        # visualize_strategies(
        #     df=df,
        #     strategy_keys=["mean~20", "mean~10"],
        #     y_name=metric_name,
        #     path=current_path
        # )


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
