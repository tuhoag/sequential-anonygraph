# Time-Aware Publishing of Knowledge Graphs
This repository contains the source code of the paper [Time-Aware Anonymization of Knowledge Graphs](https://dl.acm.org/doi/10.1145/3563694). The paper creates a privacy-preserving technology for sequential publishing of knowledge graphs. The technology protects users from identity and attribute leakage even when attackers exploit all published versions of a knowledge graph.

## Installation
This repository requires the following program to be installed:
- Python 3: [https://www.python.org/downloads/](https://www.python.org/downloads/)
- Anaconda/Miniconda: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

Then, you can install the required packages for this repository by running the following command: `conda env create -f environment.yml`. The command creates a conda environment named `anonygraph` containing all the required packages.

## Structure
This repository is organized into the following folders and files:
- `anonygraph`: the primary source code of the project.
  - `algorithms`: contains all algorithms,
  - `assertions`: consists of codes to validate the algorithms' outputs,
  - `data`: loads original datasets to our format. Each dataset has a class to read their raw data and convert them to our format. `dynamic_graph.py`, `subgraph.py` contains the code to load datasets converted in our format.
  - `evaluation`: evaluates the quality of algorithms. This includes the metrics to measure the quality of clusters, anonymized knowledge graphs. It also contains the code to train the Relational Graph Convolution Network for the node classification task.
  - `info_loss`: measures the information loss between users in the original knowledge graphs.
  - `time_graph_generators`: simulates various scenarios to split data to different snapshots.
  - `utils`: contains some functions that can be used in various parts of the projects.
- `data`: contains the raw datasets
- `exp_data`: the experimental results
- `outputs`: store the algorithms' outputs
- `scripts`: contains the some shell scripts to run the repository.
- `tests`: contains some unit tests.
- `*.py`: the Python files in the root folder are APIs to interact with our repository.
## Datasets
Datasets in this project are downloaded from the following sources:
 - Stanford SNAP: Email-Eu-core, Email-temp, Google+.
 - Multimodal Knowledge Graphs (https://github.com/nle-ml/mmkb): ICEWS14, ICEWS0515, Yago15.


## Data Generation
Datasets are put in `data` folder. To use a dataset, you must follow three steps: raw knowledge graph generation, k values generation, and distance matrix generation.

### Import Raw Datasets
To import a dataset, you can use `generate_raw_dynamic_graph.py`. For instance, to import the email-temp dataset, you can execute the following command:

<code>python generate_raw_dynamic_graph.py --data=email-temp</code>

Supported datasets can be found in `load_graph_from_raw_data` function at `anonygraph/utils/data.py`.

### Generate Snapshots
We have many strategies to simulate snapshots from an imported dataset. Supported strategies can be found in `anonygraph/time_graph_generators/__init__.py`.

For example, to generate `2` snapshots from `email-temp` with `mean` strategy (i.e., snapshots have similar number of edges), you must first generate a timestamp file using `mean` strategy. `mean` groups `email-temp`'s timestamps into `2` timestamps such that `2` snapshots satisfy `mean`'s description. The generation can be done with the following command:

<code>python generate_time_groups.py --data=email-temp --strategy=mean --n_sg=2 --log=d</code>

Then, you have to indicate which relation is the sensitive one and is required to be protected.

<code>python generate_svals.py --data=email-temp --strategy=mean --n_sg=2</code>

Now, you can generate `2` snapshots and the distance matrix storing the information loss between users in the snapshots by using the following command:

<code>python run_generate_sequence_subgraphs_and_pair_dist.py --data=email-temp --strategy=mean --n_sg=2</code>

### Anonymize Snapshots
The anonymization has three steps: clusters generation, knowledge graph generalization, and history updates. The steps are illustrated in `anonymize.py` which calls `anonymize_clusters.py`, `anonymize_kg.py` and `anonymize_history.py`. I created an API to make it easier to generate all

You can run Clusters Generation for all snapshots of a simulation setting (e.g., `mean` with `2` snapshots) with the following command:

<code>python run_anonymization.py --data=email-temp --strategy=mean --n_sg=2 --calgo=km --enforcer=gs --galgo=ad2 --k_list=2,4,6,8,10 --l_list=1,2,3,4 --anony_mode=all --run_mode=normal  --workers=1</code>

Here, `calgo=km` indicates that the clustering algorithm is k-Medoids. `enforcer=gs` is the MergeSplit algorithm illustrated in the paper. `galgo=ad2` is the generalization algorithm. `k_list` and `l_list` indicate the list of k and l values that you want to run. `anony_mode=all` indicates that all steps must be run. `run_mode` specifies how k and l should be generated from `k_list` and `l_list`. If `run_mode=normal`, it fixes `l` to the minimum value in `l_list` and run with all k values in `k_list`. Then, it fixes `k` to the maximum value in `k_list` and run with all l values in `l_list`. `workers=1` indicates that you only want to use one process to run the evaluation. If you increase `workers`, many settings can be executed at the same time. All the outputs will be put in `outputs` folder.

### Gather Anonymized Snapshots Quality
Python files `visualize_*.py` contain codes to measure the quality of anonymized snapshots and visualize them.

To gather and visualize the quality of anonymized snapshots, you can execute the command:

<code>python visualize_graphs_quality.py --data=email-temp --strategy=mean --n_sg=2 --refresh=y --type=fig</code>

To gather the classification accuracy of RGCN models trained on the snapshots, you should use `train.py` to train the models.

<code>python train.py --data=email-temp --strategy=mean --n_sg=2 --d_list=2 --k_list=2 --l_list=1 --max_dist_list=1 --calgo_list=km --anony=y --reset_w_list=-1 --enforcer=gs --galgo=ad2 --device=cpu --anony_mode=all</code>

Here, `d_list` is the size of embeddings. `max_dist_list` is the list of maximum distances in MergeSplit Algorithm. `calgo_list` is the list of clustering algorithms.`anony` can be either `y` or `n` to train on the anonymized or original snapshots.

Then, you can visualize the classification accuracy with the command:

<code>python visualize_training_data.py --data=email-temp --strategy=mean --n_sg=2</code>

## Contact
If you are interested in the repository and want to discuss, feel free to contact me at [contact@tuhoang.me](mailto:contact@tuhoang.me). If you use this repository in your project, please cite our paper.

<pre><code>@article{10.1145/3563694,
author = {Hoang, Anh-Tu and Carminati, Barbara and Ferrari, Elena},
title = {Time-Aware Anonymization of Knowledge Graphs},
year = {2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {2471-2566},
url = {https://doi.org/10.1145/3563694},
doi = {10.1145/3563694},
journal = {ACM Transactions on Privacy and Security},
month = {sep},
keywords = {Knowledge Graphs, Anonymization, Privacy}
}</code></pre>
