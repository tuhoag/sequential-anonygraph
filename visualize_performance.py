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

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
import anonygraph.utils.visualization as visual
import anonygraph.time_graph_generators as generators
import anonygraph.algorithms.clustering as calgo
import anonygraph.algorithms as algo
import anonygraph.evaluation.clusters_metrics as cmetrics
import anonygraph.constants as constants

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_sequence_data_argument(parser)
    rutils.add_workers_argument(parser)
    rutils.add_log_argument(parser)

    parser.add_argument("--anony_mode")
    parser.add_argument("--refresh", type=rutils.str2bool)


def get_stats_from_path(stats_path):
    stats = pd.read_csv(stats_path)
    info = putils.extract_info_from_stats_path(stats_path)

    stats_info = stats.to_dict(orient='records')
    for stat_item in stats_info:
        stat_item.update(info)

    # raise Exception(stats_info)

    # quality_info = get_stats_quality(clusters, subgraph, info)
    # info.update(quality_info)

    return stats_info


def prepare_data(data_info, num_workers, args):
    data_name, sample, strategy, anony_mode = data_info["data"], data_info[
        "sample"], data_info["strategy"], data_info["anony_mode"]

    stats_dir_path = putils.get_all_performance_stats_path(
        data_name, sample, strategy, anony_mode, args
    )
    logger.debug("stats path: {}".format(stats_dir_path))

    stats_paths = glob.glob(stats_dir_path + "/*")
    logger.info("preparing data from {} stats".format(len(stats_paths)))
    logger.debug(stats_paths)

    start_time = time()
    raw_data = list(
        itertools.chain.from_iterable(
            Parallel(n_jobs=num_workers)(
                delayed(get_stats_from_path)(path)
                for path in tqdm(stats_paths)
            )
        )
    )
    logger.debug(raw_data)
    # raise Exception(raw_data)
    logger.info(
        "finished preparing data in {} seconds".format(time() - start_time)
    )

    return raw_data


def add_more_info(df):
    df["clustering_and_history"] = df["clustering"] + df["gen_history"]
    df["clustering_and_gen"] = df["clustering"] + df["gen_kg"]
    df["all"] = df["clustering"] + df["gen_history"] + df["gen_kg"]


def visualize_performance(
    df, w_values, k_values, max_dist_values, t_values, y_name, x_name, cat_name
):
    logger.debug(df)
    df.sort_values(by=["w", "k", "t"], inplace=True)
    df = df[
        (df["w"].isin(w_values))
        & (df["k"].isin(k_values))
        & (df["max_dist"].isin(max_dist_values))
        & (df["t"].isin(t_values))
    ]

    logger.info("visualizing data: {}".format(df))
    w_values = df["w"].unique()
    t_values = df["t"].unique()
    k_values = df["k"].unique()
    max_dist_values = df["max_dist"].unique()
    cat_values = df[cat_name].unique()
    x_values = df[x_name].unique()

    logger.debug("w values: {}".format(w_values))
    logger.debug("t values: {}".format(t_values))
    logger.debug("k values: {}".format(k_values))
    logger.debug("max_dist values: {}".format(max_dist_values))
    logger.debug("{} values: {}".format(cat_name, cat_values))

    num_cat_values = len(cat_values)

    current_palette = sns.color_palette(n_colors=num_cat_values, palette="bright")
    sns.lineplot(
        data=df,
        x=x_name,
        y=y_name,
        hue=cat_name,
        style=cat_name,
        markers=True,
        palette=current_palette
    )
    plt.ylabel("time (seconds)")
    plt.grid(linestyle="--")
    plt.xticks(x_values)
    plt.show()

def visualize_genkg_performance(df):
    new_df = df[["k", "w", "t", "galgo", "clustering", "gen_kg"]]
    logger.info(new_df)

    ad_df = new_df[
        (df["galgo"]=="ad")
    ]
    logger.info("ad df: \n{}".format(ad_df))

    ad2_df = new_df[
        (df["galgo"]=="ad2")
    ]

    logger.info("ad2 df: \n{}".format(ad2_df))

    merged_df = ad_df.merge(ad2_df, on=["w", "t", "k"], how="inner", suffixes=("_ad", "_ad2")).sort_values(["w", "k", "t"])

    logger.info(merged_df)
    logger.info("{}".format(merged_df.columns))
    logger.info("{}".format(merged_df[["w", "k", "t", "clustering_ad", "clustering_ad2", "gen_kg_ad", "gen_kg_ad2"]]))

    new_df = new_df[
        (new_df["w"] == 1)
        &(new_df["t"] <= 20)
    ]
    sns.lineplot(data=new_df, x="t", y="gen_kg", style="galgo", hue="k")
    plt.show()

def calculate_mean_performance(df):
    df = df[
        (df["galgo"]=="ad2")
        &(df["t"] <= 30)
    ]
    new_df = df.groupby(by=["l", "k"], as_index=False).agg({
        "clustering_and_gen": ["mean"],
        "all": ["mean"],
        "clustering": ["mean"],
        "gen_kg": ["mean"],
        "t": ["max"]
    })
    # logger.debug(new_df)
    # raise Exception()
    logger.debug(new_df.columns.values)

    columns = []
    for col in new_df.columns.values:
        new_cols = list(filter(lambda item: item != "", col))
        logger.debug(new_cols)

        columns.append(".".join(new_cols))

    new_df.columns = columns
    new_df.reset_index(inplace=True)

    return new_df

def visualize_mean_performance(
    df, w_values, reset_w_values, k_values, l_values, max_dist_values, calgo_values, x_name
):
    logger.debug(df.columns)

    df.sort_values(by=["w", "k", "l", "t"], inplace=True)
    df = df[
        (df["w"].isin(w_values))
        & (df["reset_w"].isin(reset_w_values))
        & (df["k"].isin(k_values))
        & (df["max_dist"].isin(max_dist_values)
        &  (df["l"].isin(l_values)))
        & (df["calgo"].isin(calgo_values))
        & (df["enforcer"].isin(["gs"]))
    ]

    logger.debug("reset_w:{}".format(df["reset_w"].unique()))
    logger.debug("w:{}".format(df["w"].unique()))
    logger.debug("k:{}".format(df["k"].unique()))
    logger.debug("l:{}".format(df["l"].unique()))
    logger.debug("max_dist:{}".format(df["max_dist"].unique()))
    logger.debug("calgo:{}".format(df["calgo"].unique()))
    logger.debug("enforcer:{}".format(df["enforcer_str"].unique()))
    logger.debug("df:{}".format(df[["calgo", "k", "l", "max_dist", "t"]]))
    # logger.debug(df.head)
    # new_df = calculate_mean_performance(df)
    logger.debug(len(df["t"].tolist()))
    logger.debug(df["t"].tolist())

    new_df = df.groupby(by=[x_name], as_index=False).agg({
        "clustering_and_gen": ["mean"],
        "all": ["mean"],
        "clustering": ["mean"],
        "gen_kg": ["mean"],
        "t": ["max"]
    })
    logger.debug(new_df.columns)

    logger.info("visualizing data: \n{}".format(new_df[[x_name, "clustering","gen_kg", "clustering_and_gen"]]))

    plt.bar(new_df[x_name], new_df['gen_kg']['mean'], bottom=new_df['clustering']['mean'], label='Generalization')
    plt.bar(new_df[x_name], new_df['clustering']['mean'], label='Clusters Generation')

    x_values = df[x_name].unique()

    plt.ylabel("Execution Time (seconds)")
    plt.xlabel(visual.get_title(x_name))
    # plt.grid(linestyle="--")
    plt.xticks(visual.get_xticks(x_name, x_values, 8))
    plt.legend()

    data_name = df["data"].unique()[0]
    for ext in ["png", "pdf"]:
        path = os.path.join("exp_data", "performance", "{}_{}.{}".format(data_name,x_name, ext))

        plt.savefig(path)

    plt.show()
    plt.clf()
    # raise Exception()
    # y_name = "{}.mean".format(y_name)

    # cat_values = new_df[cat_name].unique()
    # x_values = new_df[x_name].unique()
    # y_values = new_df[y_name].unique()

    # logger.info("x ({}): {}".format(x_name, x_values))
    # logger.info("y ({}): {}".format(y_name, y_values))
    # logger.info("cat ({}): {}".format(cat_name, cat_values))

    # num_cat_values = len(cat_values)

    # current_palette = sns.color_palette(n_colors=num_cat_values, palette="bright")
    # sns.lineplot(
    #     data=new_df,
    #     x=x_name,
    #     y=y_name,
    #     hue=cat_name,
    #     style=cat_name,
    #     markers=True,
    #     palette=current_palette
    # )
    # plt.ylabel("time (seconds)")
    # plt.grid(linestyle="--")
    # plt.xticks(x_values)
    # plt.show()



def main(args):
    logger.debug(args)
    data_path = putils.get_performance_stats_exp_data_path(
        args["data"], args["sample"], args["strategy"], args["anony_mode"], args
    )

    df = visual.get_exp_data(
        exp_path=data_path,
        prepare_data_fn=prepare_data,
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

    visualize_mean_performance(
        df=df,
        w_values=[-1],
        reset_w_values=[-1],
        k_values=[2, 4, 6, 8, 10],
        l_values=[1],
        max_dist_values=[1.0],
        calgo_values=["km"],
        x_name="k",
        # y_name="clustering_and_gen",
        # cat_name="k"
    )

    visualize_mean_performance(
        df=df,
        w_values=[-1],
        reset_w_values=[-1],
        k_values=[10],
        l_values=[1,2,3,4],
        max_dist_values=[1.0],
        calgo_values=["km"],
        x_name="l",
    )



if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
