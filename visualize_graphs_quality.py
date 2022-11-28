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

import anonygraph.utils.visualization as vutils
import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
import anonygraph.time_graph_generators as generators
import anonygraph.algorithms.clustering as calgo
import anonygraph.algorithms as algo
import anonygraph.evaluation.subgraphs_metrics as metrics
from anonygraph.constants import *

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_sequence_data_argument(parser)
    parser.add_argument("--anony_mode")
    rutils.add_workers_argument(parser)
    rutils.add_log_argument(parser)

    parser.add_argument("--type")


    parser.add_argument("--refresh", type=rutils.str2bool)

def add_more_info(df):
    df["ratio_fake_edges"] = df[FAKE_EDGES_METRIC
                               ] / df[ANONYMIZED_EDGES_METRIC]
    df["ratio_fake_entities"] = df[FAKE_ENTITIES_METRIC
                                  ] / df[ANONYMIZED_ENTITIES_METRIC]
    df[r"$\tau$"] = df["max_dist"]

    df["calgo_k"] = df["calgo"] + "_" + df["k"].astype(str)

    df["enforcer_name"].replace("gs", "merge_split", inplace=True)
    df["enforcer_name"].replace("ir", "invalid_removal", inplace=True)
    df["calgo_enforcer"] = df["calgo"] + "#" + df["enforcer_name"]
    df["t"] = df["t"] + 1

    df["ratio_fake_removed_entities"] = (df[FAKE_ENTITIES_METRIC] + df[REMOVED_ENTITIES_METRIC])/df[RAW_ENTITIES_METRIC]

    df["ratio_fake_removed_edges"] = (df[FAKE_EDGES_METRIC] + df[REMOVED_EDGES_METRIC]) / df[RAW_EDGES_METRIC]

    df["reset_w_name"] = df["reset_w"]
    df.loc[df["reset_w"] == -1, "reset_w_name"] = "No Reset"

    logger.debug(df["reset_w_name"].unique())
    # raise Exception()


def visualize_fine_tune(
    df, w_values, k_values,l_values, max_dist_values, reset_w_values, enforcer_values, calgo_values, y_name, x_name, cat_name, path=None
):
    logger.info("visualize x: {} - y: {} - cat: {}".format(x_name, y_name, cat_name))
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
    plt.ylabel(vutils.get_title(y_name))
    plt.grid(linestyle="--")
    plt.xticks(vutils.get_xticks(x_name, x_values, 8))
    plt.legend(title=vutils.get_title(cat_name), labels=cat_values)

    if path is not None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        plt.savefig(path)
        logger.info("saved to {}".format(path))

    plt.show()
    plt.clf()

def get_name(name):

    short2full_name = {
        "km": "k-Medoids",
        "hdbscan": "HDBSCAN",
        "gs": "Merge_Split",
        "ir": "Invalid-Removal",
        "adm": "AIL",
        "radm": "RAIL",
    }

    return short2full_name[name]

def visualize_k_table(df, k_values, l_values, max_dist_values, calgo_values, reset_w_values, enforcer_values, metric_names, col_name, path):
    df = df[
        (df["k"].isin(k_values))
        & (df["l"].isin(l_values))
        & (df["max_dist"].isin(max_dist_values))
        & (df["reset_w"].isin(reset_w_values))
        & (df["calgo"].isin(calgo_values))
        & (df["enforcer"].isin(enforcer_values))
    ]

    t_modes = ["1", "2..20"]
    col_values = df[col_name].unique()
    # metric_names = ["adm", "radm"]
    with open(path, "w") as f:
        for col_value in col_values:
            current_df = df[df[col_name] == col_value]
            line_splits = [col_value]

            for t_mode in t_modes:
                if t_mode == "1":
                    t_df = current_df[current_df["t"] == 1]

                elif t_mode == "2..20":
                    t_df = current_df[current_df["t"] != 1]
                else:
                    raise Exception(t_mode)

                for metric_name in metric_names:
                    if t_mode == "1":
                        metric_value = t_df[metric_name].values[0]
                    elif t_mode == "2..20":
                        metric_value = t_df[metric_name].mean()
                    else:
                        raise Exception(t_mode)

                    line_splits.append("{:10.4f}".format(metric_value))
                    logger.debug("{}, {}, {}, {}".format(col_value, t_mode, metric_name, metric_value))

            logger.debug(line_splits)
            line_str = " & ".join(map(str, line_splits))
            f.write("{}\\\\ \n".format(line_str))



def visualize_calgo_table(df, path):
    w_values = [-1]
    reset_w_values = [-1]
    k_values = [2]
    l_values = [1]
    max_dist_values = [1]
    enforcer_values = ["ir", "gs"]
    calgo_values = ["km", "hdbscan"]

    metric_names = ["adm", "radm"]


    df.sort_values(by=["calgo", "enforcer", "k", "l", "reset_w", "max_dist", "t"], inplace=True)
    df = df[(df["w"].isin(w_values)) & (df["k"].isin(k_values)) &
            (df["max_dist"].isin(max_dist_values)) & (df["l"].isin(l_values))
            & (df["reset_w"].isin(reset_w_values))
            # & (df["enforcer"].isin(enforcer_values))
            # & (df["calgo"].isin(calgo_values))
            ]

    result = ""
    t_modes = ["1", "2..20"]
    with open(path, "w") as f:
        for enforcer in enforcer_values:
            enforcer_df = df[df["enforcer"] == enforcer]

            enforcer_name = get_name(enforcer)

            enforcer_str = "\multirow{{2}}{{*}}{}".format(enforcer_name)


            for t_mode in t_modes:
                line_splits = []

                if t_mode == "1":
                    t_df = enforcer_df[enforcer_df["t"] == 1]
                    line_splits.append(enforcer_str)
                elif t_mode == "2..20":
                    t_df = enforcer_df[enforcer_df["t"] != 1]
                    line_splits.append("")
                else:
                    raise Exception("Unsupported t_mode: {}".format(t_mode))

                line_splits.append(t_mode)

                for calgo in calgo_values:
                    calgo_name = get_name(calgo)
                    calgo_df = t_df[t_df["calgo"] == calgo]

                    for metric_name in metric_names:
                        metric_full_name = get_name(metric_name)

                        if t_mode == "1":
                            metric_value = calgo_df[metric_name].values[0]
                        else:
                            avg_df = calgo_df[["t", metric_name]]
                            # logger.debug("avg_df (len: {}): {}".format(len(avg_df), avg_df[metric_name]))
                            metric_value = avg_df[metric_name].mean()
                            # logger.debug("avg: {}".format(avg_df[metric_name].mean()))

                        logger.debug("{} & {} & {} & {} & {}".format(enforcer_name, t_mode, calgo_name, metric_full_name,  metric_value))

                        line_splits.append("{:10.4f}".format(metric_value))

                logger.debug(line_splits)
                line_str = " & ".join(map(str, line_splits))
                logger.debug(line_str)
                f.write("{}\n".format(line_str))

    print(result)

def visualize_figures(df, strategy_str, metric_names, fig_dir_path):
    for metric in metric_names:
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
            cat_name="reset_w_name",
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

def visualize_tables(df, strategy_str, metric_names, fig_dir_path):
    # col_names = ["k", "l", "max_dist", "reset_w"]
    df.sort_values(by=["calgo", "enforcer", "k", "l", "reset_w", "max_dist", "t"], inplace=True)

    fig_path = os.path.join(fig_dir_path, "{}-calgo.tex".format(strategy_str))
    visualize_calgo_table(df, fig_path)
    fig_path = os.path.join(fig_dir_path, "{}-k.tex".format(strategy_str))
    visualize_k_table(
        df=df,
        k_values=[2,4,6,8,10],
        l_values=[1],
        max_dist_values=[1],
        reset_w_values=[-1],
        calgo_values=["km"],
        enforcer_values=["gs"],
        metric_names=["adm", "radm"],
        col_name="k",
        path=fig_path
    )

    fig_path = os.path.join(fig_dir_path, "{}-l.tex".format(strategy_str))
    visualize_k_table(
        df=df,
        k_values=[10],
        l_values=[1, 2, 3, 4],
        max_dist_values=[1],
        reset_w_values=[-1],
        calgo_values=["km"],
        enforcer_values=["gs"],
        metric_names=["adm", "radm"],
        col_name="l",
        path=fig_path
    )

    fig_path = os.path.join(fig_dir_path, "{}-tau.tex".format(strategy_str))
    visualize_k_table(
        df=df,
        k_values=[10],
        l_values=[4],
        max_dist_values=[0, 0.25, 0.5, 0.75, 1],
        reset_w_values=[-1],
        calgo_values=["km"],
        enforcer_values=["gs"],
        metric_names=["adm", "radm"],
        col_name="max_dist",
        path=fig_path
    )

    fig_path = os.path.join(fig_dir_path, "{}-resetw.tex".format(strategy_str))
    visualize_k_table(
        df=df,
        k_values=[10],
        l_values=[4],
        max_dist_values=[1],
        reset_w_values=[-1, 1, 2, 3, 4],
        calgo_values=["km"],
        enforcer_values=["gs"],
        metric_names=["adm", "radm"],
        col_name="reset_w",
        path=fig_path
    )

def main(args):
    data_name = args["data"]
    logger.debug(args)
    data_path = putils.get_tuning_graphs_exp_data_path(
        args["data"], args["sample"], args["strategy"], args
    )

    df = vutils.get_exp_data(
        exp_path=data_path,
        prepare_data_fn=vutils.prepare_anonymized_subgraphs_data,
        prepare_data_args={
            "data": args["data"],
            "sample": args["sample"],
            "strategy": args["strategy"],
            "anony_mode": CLUSTERS_AND_GRAPH_ANONYMIZATION_MODE,
        },
        workers=args["workers"],
        refresh=args["refresh"],
        args=args
    )

    add_more_info(df)

    logger.debug(df)
    logger.info("visualizing")
    fig_dir_path = os.path.join(os.path.dirname(data_path), "figures")
    strategy_str = "{}_{}_{}".format(data_name, args["strategy"], args["n_sg"])
    logger.debug(fig_dir_path)
    metric_names = [ADM_METRIC, "ratio_fake_removed_entities", "ratio_fake_removed_edges", RADM_METRIC]

    if args["type"] == "fig":
    # if data_name == "email-temp":
        visualize_figures(df, strategy_str, metric_names, fig_dir_path)
    elif args["type"] == "tab":
        visualize_tables(df, strategy_str, ["adm", "radm"], fig_dir_path)



if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
