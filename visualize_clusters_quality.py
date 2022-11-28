import math
from time import time
from networkx.algorithms.components.strongly_connected import condensation
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

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
import anonygraph.utils.visualization as visual
import anonygraph.time_graph_generators as generators
import anonygraph.algorithms.clustering as calgo
import anonygraph.algorithms as algo
import anonygraph.evaluation.clusters_metrics as cmetrics
from anonygraph.constants import *

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_sequence_data_argument(parser)
    rutils.add_workers_argument(parser)
    rutils.add_log_argument(parser)

    parser.add_argument("--anony_mode")
    parser.add_argument("--refresh", type=rutils.str2bool)


def add_more_info(df):
    df["ratio_fake_edges"] = df[FAKE_EDGES_METRIC
                               ] / df[REAL_EDGES_METRIC]
    df["ratio_fake_entities"] = df[FAKE_ENTITIES_METRIC
                                  ] / df[REAL_ENTITIES_METRIC]

    df["ratio_intersection_entities"] = df[REAL_ENTITIES_METRIC] / df[RAW_ENTITIES_METRIC]
    df["ratio_intersection_edges"] = df[REAL_EDGES_METRIC] / df[RAW_EDGES_METRIC]

    df[r"$\tau$"] = df["max_dist"]
    df["t"] = df["t"] + 1

    df["invalid_anonymity"] = df[ANONYMIZED_ANONYMITY_METRIC] < df["k"]
    # df["ratio_big_clusters"] = df[NUM_BIG_CLUSTERS] / df[NUM_CLUSTERS]
    # df["ratio_entities_in_big_clusters"] = df[NUM_ENTITIES_IN_BIG_CLUSTERS] / df[ANONYMIZED_ENTITIES_METRIC]
    df["calgo_k"] = df["calgo"] + "_" + df["k"].astype(str)

    df["enforcer_name"].replace("gs", "merge_split", inplace=True)
    df["enforcer_name"].replace("ir", "invalid_removal", inplace=True)
    df["calgo_enforcer"] = df["calgo"] + "#" + df["enforcer_name"]



    # df["data"].replace("yago15", "yago", inplace=True)

def visualize_fine_tune(
    df, w_values, k_values,l_values, max_dist_values, reset_w_values, enforcer_values, calgo_values, y_name, x_name, cat_name, path=None
):
    logger.debug(df.columns)
    logger.debug("df (len: {}): {}".format(len(df), df))
    df.sort_values(by=["calgo", "enforcer", "k", "l", "reset_w", "max_dist", "t"], inplace=True)
    df = df[(df["w"].isin(w_values)) & (df["k"].isin(k_values)) &
            (df["max_dist"].isin(max_dist_values)) & (df["l"].isin(l_values))
            & (df["reset_w"].isin(reset_w_values))
            & (df["enforcer"].isin(enforcer_values))
            & (df["calgo"].isin(calgo_values))
            ]

    w_values = df["w"].unique()
    t_values = df["t"].unique()
    k_values = df["k"].unique()
    l_values = df["l"].unique()
    max_dist_values = df["max_dist"].unique()
    reset_w_values = df["reset_w"].unique()
    enforcer_name_values = df["enforcer_name"].unique()
    calgo_names = df["calgo"].unique()
    x_values = df[x_name].unique()
    calgo_k_values = df["calgo_k"].unique()
    cat_values = df[cat_name].unique()

    logger.debug("visualizing filtered df (len: {}): {}".format(len(df), df[[x_name, y_name, cat_name]]))
    logger.debug("w values: {}".format(w_values))
    logger.debug("t values: {}".format(t_values))
    logger.debug("k values: {}".format(k_values))
    logger.debug("l values: {}".format(l_values))
    logger.debug("max_dist values: {}".format(max_dist_values))
    logger.debug("reset_w values: {}".format(reset_w_values))
    logger.debug("enforcer values: {}".format(enforcer_name_values))
    logger.debug("calgo_names: {}".format(calgo_names))
    logger.debug("calgo_k_values: {}".format(calgo_k_values))
    num_cat_values = len(df[cat_name].unique())

    current_palette = sns.color_palette(n_colors=num_cat_values, palette="bright")
    sns.lineplot(
        data=df, x=x_name, y=y_name,
        hue=cat_name,
        style=cat_name,
        palette=current_palette,
        legend=False,
    )
    plt.ylabel(visual.get_title(y_name))
    plt.grid(linestyle="--")
    plt.xticks(visual.get_xticks(x_name, x_values, 8))
    plt.legend(title=visual.get_title(cat_name), labels=cat_values)

    # plt.xticks(set(range(min(t_values), max(t_values), math.floor(max(t_values) / 10))).union(set([max(t_values)])))


    if path is not None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        plt.savefig(path)

    plt.show()
    plt.clf()



def visualize_df_stats(df):
    k_values = df["k"].unique()
    l_values = df["l"].unique()
    w_values = df["w"].unique()
    invalid_values = df["invalid_anonymity"]
    print(df[["k", ANONYMIZED_ANONYMITY_METRIC, "invalid_anonymity"]])

    print("k values: {}".format(k_values))
    print("l values: {}".format(l_values))
    print("w values: {}".format(w_values))
    print("invalid values: {}".format(sum(invalid_values)))

def main(args):
    logger.debug(args)
    data_name = args["data"]
    data_path = putils.get_tuning_clusters_exp_data_path(
        args["data"], args["sample"], args["strategy"], args["anony_mode"], args
    )

    df = visual.get_exp_data(
        exp_path=data_path,
        prepare_data_fn=visual.prepare_clusters_data,
        prepare_data_args={
            "data": args["data"],
            "sample": args["sample"],
            "strategy": args["strategy"],
            "anony_mode": args["anony_mode"],
        },
        workers=args["workers"],
        refresh=args["refresh"],
        args=args
    )

    add_more_info(df)

    logger.debug(df)
    logger.info("visualizing")
    visualize_df_stats(df)



    metrics = [
        "adm",
        # "radm",
        # "ratio_intersection_entities",
        # "ratio_entities_in_big_clusters",
        # "ratio_big_clusters",
        # "ratio_fake_entities",
        # ANONYMIZED_ANONYMITY_METRIC,
    ]
    fig_dir_path = os.path.join(os.path.dirname(data_path), "figures")
    strategy_str = "{}_{}_{}".format(data_name, args["strategy"], args["n_sg"])
    logger.debug(fig_dir_path)
    for metric in metrics:
        fig_path = os.path.join(fig_dir_path, "{}-{}-t-calgo_enforcer.pdf".format(strategy_str, metric))
        visualize_fine_tune(
            df=df,
            w_values=[-1],
            k_values=[2],
            l_values=[1],
            max_dist_values=[1],
            reset_w_values=[-1],
            enforcer_values=[GREEDY_SPLIT_ENFORCER, INVALID_REMOVAL_ENFORCER],
            calgo_values=["km", "hdbscan"],
            x_name="t",
            y_name=metric,
            cat_name="calgo_enforcer",
            path=fig_path
        )

        fig_path = os.path.join(fig_dir_path, "{}-{}-t-calgo.pdf".format(strategy_str, metric))
        visualize_fine_tune(
            df=df,
            w_values=[-1],
            k_values=[2],
            l_values=[1],
            max_dist_values=[1],
            reset_w_values=[-1],
            enforcer_values=[GREEDY_SPLIT_ENFORCER],
            calgo_values=["km", "hdbscan"],
            x_name="t",
            y_name=metric,
            cat_name="calgo",
            path=fig_path
        )

        fig_path = os.path.join(fig_dir_path, "{}-{}-t-k.pdf".format(strategy_str, metric))
        visualize_fine_tune(
            df=df,
            w_values=[-1],
            k_values=[2, 4, 6, 8, 10],
            l_values=[1],
            max_dist_values=[1],
            reset_w_values=[-1],
            enforcer_values=[GREEDY_SPLIT_ENFORCER],
            calgo_values=["km"],
            x_name="t",
            y_name=metric,
            cat_name="k",
            path=fig_path
        )

        fig_path = os.path.join(fig_dir_path, "{}-{}-t-l.pdf".format(strategy_str, metric))
        visualize_fine_tune(
            df=df,
            w_values=[-1],
            k_values=[10],
            l_values=[1, 2, 3, 4],
            max_dist_values=[1],
            reset_w_values=[-1],
            enforcer_values=[GREEDY_SPLIT_ENFORCER],
            calgo_values=["km"],
            x_name="t",
            y_name=metric,
            cat_name="l",
            path=fig_path
        )


        fig_path = os.path.join(fig_dir_path, "{}-{}-t-resetw.pdf".format(strategy_str, metric))
        visualize_fine_tune(
            df=df,
            w_values=[-1],
            k_values=[10],
            l_values=[4],
            max_dist_values=[1],
            reset_w_values=[-1, 1, 2, 4, 5],
            enforcer_values=[GREEDY_SPLIT_ENFORCER],
            calgo_values=["km"],
            x_name="t",
            y_name=metric,
            cat_name="reset_w",
            path=fig_path
        )


        fig_path = os.path.join(fig_dir_path, "{}-{}-t-tau.pdf".format(strategy_str, metric))
        visualize_fine_tune(
            df=df,
            w_values=[-1],
            k_values=[10],
            l_values=[4],
            max_dist_values=[0, 0.25, 0.5, 0.75, 1],
            reset_w_values=[-1],
            enforcer_values=[GREEDY_SPLIT_ENFORCER],
            calgo_values=["km"],
            x_name="t",
            y_name=metric,
            cat_name="max_dist",
            path=fig_path
        )


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
