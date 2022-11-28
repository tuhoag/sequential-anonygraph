import numpy as np
import torch
import logging

import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import DGLDataset
import os
import itertools

logger = logging.getLogger(__name__)

def get_entity_idx(entity_id2idx, entity_id, create_new=True):
    entity_idx = entity_id2idx.get(entity_id, None)
    if entity_idx is None and create_new:
        entity_idx = len(entity_id2idx)
        entity_id2idx[entity_id] = entity_idx

    return entity_idx

def load_rels(rels_iter, entity_id2idx):
    data_dict = {}

    attrs_entity_ids = set(entity_id2idx.keys())
    existed_entity_ids = set()

    count = 0
    for edge in rels_iter:
        count += 1
        entity1_id, relation_id, entity2_id = edge.strip().split(",")

        relation_str = str(relation_id)
        key = ("user", relation_str, "user")
        relation_edges = data_dict.get(key, None)
        if relation_edges is None:
            relation_edges = ([], [])
            data_dict[key] = relation_edges

        entity1_idx = get_entity_idx(entity_id2idx, entity1_id)
        entity2_idx = get_entity_idx(entity_id2idx, entity2_id)

        relation_edges[0].append(entity1_idx)
        relation_edges[1].append(entity2_idx)

        existed_entity_ids.add(entity1_id)
        existed_entity_ids.add(entity2_id)

    logger.debug("processed {} edges".format(count))
    # if len(attrs_entity_ids) != len(existed_entity_ids):
    #     logger.debug("adding fake edges")
    #     logger.info(len(attrs_entity_ids))
    #     logger.info(len(existed_entity_ids))
    #     remaining_ids = attrs_entity_ids.difference(existed_entity_ids)
    #     logger.debug(len(remaining_ids))

    #     count = 0
    #     relation_edges = ([],[])
    #     for entity1_id, entity2_id in itertools.product(remaining_ids, remaining_ids):
    #         if entity2_id != entity1_id:
    #             count += 1
    #             entity1_idx = get_entity_idx(entity_id2idx, entity1_id)
    #             entity2_idx = get_entity_idx(entity_id2idx, entity2_id)

    #             # key = ("user", "fake", "user")

    #             relation_edges[0].append(entity1_idx)
    #             relation_edges[0].append(entity2_idx)

    #     data_dict[("user", "fake", "user")] = relation_edges

    #     logger.debug(count)

        # raise Exception()

    for key in data_dict:
        relation_edges = data_dict[key]
        data_dict[key] = (torch.tensor(relation_edges[0]), torch.tensor(relation_edges[1]))

    return data_dict

def load_attributes(attrs_iter, entity_id2idx):
    if entity_id2idx is None:
        raise Exception("entity_id2idx is None")

    value_idx2entity_idxes = {}
    value2value_idx = {}
    count = 0
    for line in attrs_iter:
        count += 1
        entity_id, relation_id, value_id = line.strip().split(",")

        entity_idx = get_entity_idx(entity_id2idx, entity_id, False)
        if entity_idx is None:
            continue

        val_key = (relation_id, value_id)
        val_idx = get_entity_idx(value2value_idx, val_key)

        entity_idxes = value_idx2entity_idxes.get(val_idx, None)
        if entity_idxes is None:
            entity_idxes = []
            value_idx2entity_idxes[val_idx] = entity_idxes

        entity_idxes.append(entity_idx)

    num_entities = len(entity_id2idx)
    logger.debug("num entities: {}".format(num_entities))

    if count == 0:
        num_features = 1
        for i in range(num_features):
            get_entity_idx(value2value_idx, i)

        features_data = np.ones((num_entities, num_features),dtype=np.double)
        logger.debug("features data shape: {}".format(features_data.shape))
    else:
        num_features = len(value_idx2entity_idxes)
        features_data = np.zeros((num_entities, num_features),dtype=np.double)
        logger.debug("features data shape: {}".format(features_data.shape))
        for value_idx, entity_idxes in value_idx2entity_idxes.items():
            for entity_idx in entity_idxes:
                features_data[entity_idx,value_idx] = 1

    features_tensors = torch.tensor(features_data, dtype=torch.float32)
    return features_tensors,value_idx2entity_idxes, value2value_idx
    # hetero_graph.nodes['user'].data['feature'] = torch.randn(n_users, n_hetero_features)

def load_labels(labels_iter, entity_id2idx):
    if entity_id2idx is None:
        raise Exception("entity_id2idx is None")

    label2entity_idxes = {}
    label_id2idxes = {}

    for line in labels_iter:
        entity_id, label_id = line.strip().split(",")
        entity_idx = get_entity_idx(entity_id2idx, entity_id, False)

        if entity_idx is None:
            continue

        label_idx = get_entity_idx(label_id2idxes, label_id)

        entity_idxes = label2entity_idxes.get(label_idx)
        if entity_idxes is None:
            entity_idxes = []
            label2entity_idxes[label_idx] = entity_idxes

        entity_idxes.append(entity_idx)

    labels_data = np.zeros((len(entity_id2idx), len(label2entity_idxes)))
    for label_idx, entity_idxes in label2entity_idxes.items():
        for entity_idx in entity_idxes:
            labels_data[entity_idx, label_idx] = 1

    labels_tensor = torch.tensor(labels_data)
    return labels_tensor, label2entity_idxes, label_id2idxes


class DummyGraph:
    def __init__(self):
        self.graph = None
        self.entity_id2idx = {}



    def load(self):
        edges_list = [
            (20, 18, 629),
            (629, 18, 20),
            (22, 11, 8595),
            (23, 11, 50),
            (24, 11, 51)
        ]

        attrs_list = [
            (20,0,1),
            (20,0,2048),
            (629,0,135),
            (22,0,31),
            (8595,0,60),
            (20,0,60),
            (0,4,1251),
            (0,6,8343),
        ]

        labels_list = [
            (20,38),
            (20,362),
            (629,362),
            (22,29),
            (8595,11),
            (10,362),
            (10,11),
        ]

        edges_data_dict, self.entity_id2idx = load_rels(edges_list)
        features_data, self.value_idx2entity_idxes, self.value2value_idx = load_attributes(attrs_list, self.entity_id2idx)
        labels_data, self.label2entity_idxes, self.label_id2idxes = load_labels(labels_list, self.entity_id2idx)

        # print(features_data)
        # print(value_idx2entity_idxes)
        # print(value2value_idx)
        # print(labels_data)
        # print(label2entity_idxes)
        # print(label_id2idxes)
        self.graph = dgl.heterograph(edges_data_dict)
        self.graph.nodes["user"].data["feature"] = features_data
        self.graph.nodes["user"].data["label"] = labels_data

        # n_users = len(self.entity_id2idx)
        # n_user_classes = 3
        # sample_labels = torch.randint(0, 2, (n_users,n_user_classes))
        # self.graph.nodes["user"].data["label"] = labels_data

        print(self.graph.nodes["user"].data["feature"])
        print(self.graph.nodes["user"].data["label"])
        # hetero_graph.nodes['user'].data['feature'] = torch.randn(n_users, n_hetero_features)
        # hetero_graph.nodes['user'].data['label'] = torch.randint(0, n_user_classes, (n_users,))

# def load_entity2idx(attrs_iter):
#     for line in attrs_iter:
#         entity_id, relation_id, value_id = line.strip().split(",")




class AnonyGraph:
    def __init__(self):
        self.graph = None
        self.entity_id2idx = {}

    def load(self, path):
        rels_path = os.path.join(path, "rels.edges")
        attrs_path = os.path.join(path, "attrs.edges")
        labels_path = os.path.join(path, "sensitive.vals")

        logger.debug("loading from: {}".format(path))
        logger.info(len(self.entity_id2idx))

        logger.debug("loading rels from: {}".format(rels_path))
        with open(rels_path, "r") as rels_iter:
            edges_data_dict = load_rels(rels_iter, self.entity_id2idx)

        logger.info(len(self.entity_id2idx))

        logger.debug("loading attrs from: {}".format(attrs_path))
        with open(attrs_path, "r") as attrs_iter:
            features_data, self.value_idx2entity_idxes, self.value2value_idx = load_attributes(attrs_iter, self.entity_id2idx)

        logger.info(len(self.entity_id2idx))
        with open(labels_path, "r") as labels_iter:
            next(labels_iter)
            labels_data, self.label2entity_idxes, self.label_id2idxes = load_labels(labels_iter, self.entity_id2idx)

        # logger.debug(edges_data_dict)




        # print(features_data)
        # print(value_idx2entity_idxes)
        # print(value2value_idx)
        # print(labels_data)
        # print(label2entity_idxes)
        # print(label_id2idxes)
        # logger.info(edges_data_dict)

        self.graph = dgl.heterograph(edges_data_dict)

        logger.debug(self.graph)
        logger.debug(features_data.shape)
        logger.debug(features_data)

        self.graph.nodes["user"].data["feature"] = features_data
        self.graph.nodes["user"].data["label"] = labels_data

        # try:
        #     self.graph = dgl.heterograph(edges_data_dict)
        #     self.graph.nodes["user"].data["feature"] = features_data
        #     self.graph.nodes["user"].data["label"] = labels_data

        #     logger.debug(self.graph)
        #     raise Exception()
        # except:
        #     self.graph = None




        # n_users = len(self.entity_id2idx)
        # n_user_classes = 3
        # sample_labels = torch.randint(0, 2, (n_users,n_user_classes))
        # self.graph.nodes["user"].data["label"] = labels_data

        # print(self.graph.nodes["user"].data["feature"])
        # print(self.graph.nodes["user"].data["label"])
        # hetero_graph.nodes['user'].data['feature'] = torch.randn(n_users, n_hetero_features)
        # hetero_graph.nodes['user'].data['label'] = torch.randint(0, n_user_classes, (n_users,))