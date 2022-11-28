import time
import datetime
import os
import logging
import re

from .dynamic_graph import DynamicGraph

logger = logging.getLogger(__name__)

def extract_nodes(file_path):
    nodes_path = os.path.join(os.path.dirname(file_path), "nodes.txt")
    relations_path = os.path.join(os.path.dirname(file_path), "relations.txt")

    with open(file_path, "r") as f, open(nodes_path, "w") as f_out, open(relations_path, "w") as f_rout:
        nodes = set()
        relations = set()

        nodes_count_dict = {}
        relations_count_dict = {}

        for line in f:
            splits = line.strip().split("\t")
            node1, relation, node2, t = splits
            nodes.add(node1)
            nodes.add(node2)
            nodes_count_dict[node2] = nodes_count_dict.get(node2, 0) + 1
            nodes_count_dict[node1] = nodes_count_dict.get(node1, 0) + 1

            relations.add(relation)
            relations_count_dict[relation] = relations_count_dict.get(relation, 0) + 1

        sorted_nodes = sorted([(node, nodes_count_dict[node]) for node in nodes], key=lambda item: -item[1])
        sorted_relations = sorted([(relation, relation_count) for (relation, relation_count) in relations_count_dict.items()], key=lambda item: -item[1])

        logger.debug("sorted_relations (len: {}): {}".format(len(sorted_relations), sorted_relations))
        for node, node_count in sorted_nodes:
            f_out.write("{}\n".format(node))

        for relation, relation_count in sorted_relations:
            f_rout.write("{}\n".format(relation))

def load_entity_names(file_path):
    entity_names = set()

    with open(file_path, "r") as f:
        for line in f:
            splits = line.strip().split(",")

            flag = splits[-1]
            node = ",".join(splits[0:-1])

            if flag == "0":
                entity_names.add(node)

    return entity_names

def is_attribute_edge(node1, relation, node2, entity_names):
    return node1 in entity_names and node2 not in entity_names

def is_relationship_edge(node1, relation, node2, entity_names):
    return node1 in entity_names and node2 in entity_names

def extract_edges_from_entity_names(file_path, graph):
    entity_names = load_entity_names(os.path.join(os.path.dirname(file_path), "nodes.txt"))


    with open(file_path, "r") as f:
        for line in f:
            splits = line.strip().split("\t")
            node1, relation, node2, t_str = splits
            t = int(time.mktime(datetime.datetime.strptime(t_str, "%Y-%m-%d").timetuple()))
            # raise Exception("{} -> {}".format(t_str, t))
            # edges.append((node1, relation, node2, t))

            if is_attribute_edge(node1, relation, node2, entity_names):
                graph.add_attribute_edge(node1, relation, node2, t)
            elif is_relationship_edge(node1, relation, node2, entity_names):
                graph.add_relationship_edge(node1, relation, node2, t)


def load_users_relationship(graph, file_path):
    logger.info("extracting edges")
    extract_nodes(file_path)
    # extract_edges_from_entity_names(file_path, graph)

    return graph


class ICEWS14Graph(DynamicGraph):
    def __init__(self):
        super().__init__()

    @staticmethod
    def from_raw_file(data_dir, args):
        # load attributes
        # load relationships
        graph = ICEWS14Graph()
        user_relationship_path = os.path.join(data_dir, 'icews_2014_train.txt')

        find_users_keywords(user_relationship_path)
        # load_users_relationship(graph, user_relationship_path)

        return graph


def load_non_users_keywords(path):
    non_users_keywords = set()

    if not os.path.exists(path):
        return non_users_keywords

    with open(path, "r") as f:
        for line in f:
            non_users_keywords.add(line.strip())

    return non_users_keywords

def load_found_usernames(path):
    names_set = set()
    with open(path, "r") as f:
        for line in f:
            name = line.strip()
            names_set.add(name)

    return names_set

class NonUserNameDetector():
    def __init__(self, non_users_keywords_path, usernames_path):
        self.non_users_keywords_path = non_users_keywords_path
        self.usernames_path = usernames_path
        self.non_users_keywords = load_non_users_keywords(non_users_keywords_path)
        self.names_set = load_found_usernames(usernames_path)

    def open_file(self):
        self.non_users_keywords_file = open(self.non_users_keywords_path, "a")
        self.usernames_file = open(self.usernames_path, "a")

    def close_file(self):
        self.non_users_keywords_file.close()
        self.usernames_file.close()

    @staticmethod
    def split_keywords(name):
        name = name.replace("(", "")
        name = name.replace(")", "")
        keywords = set(map(lambda item: item.strip(), name.split(" ")))

        logger.debug("name: {} - keywords: {}".format(name, keywords))
        if "" in keywords:
            keywords.remove("")
        return keywords

    def detect_username(self, name):
        if name in self.names_set:
            return 1

        return 0

    def detect(self, name):
        if self.detect_username(name) == 1:
            logger.debug("'{}' is in found user names".format(name))
            score = 0
        else:
            keywords = self.split_keywords(name)

            non_users_keywords = keywords.intersection(self.non_users_keywords)

            logger.debug("intersection: {}".format(non_users_keywords))

            score = len(non_users_keywords) / len(keywords) # the higher, the more chance it is not user name

        return score

    def update_keywords_data(self, name):
        keywords = self.split_keywords(name)

        for keyword in keywords:
            if keyword not in self.non_users_keywords:
                self.non_users_keywords_file.write("{}\n".format(keyword))
                self.non_users_keywords.add(keyword)

    def update_usernames_data(self, name):
        if not name in self.names_set:
            self.names_set.add(name)
            self.usernames_file.write("{}\n".format(name))




def find_users_keywords(path):
    non_users_keywords_path = os.path.join(os.path.dirname(path), "non_users_keywords.txt")
    usernames_path = os.path.join(os.path.dirname(path), "users.txt")
    non_user_name_detector = NonUserNameDetector(non_users_keywords_path, usernames_path)
    threshold = 0.3

    num_all_names = 6869
    num_lines = 72826
    current_line = 1
    non_user_name_detector.open_file()
    with open(path, "r") as f:
        for line in f:
            splits = line.strip().split("\t")
            node1, relation, node2, t_str = splits

            for name in [node1, node2]:
                non_user_score = non_user_name_detector.detect(name)
                logger.info("[line: {}/{} - users: {}] name: {} - non user name score: {}".format(current_line, num_lines, len(non_user_name_detector.names_set), name, non_user_score))
                if non_user_score >= threshold: # is non user name
                    logger.info("found a non user name: {}".format(name))
                    non_user_name_detector.update_keywords_data(name)
                elif non_user_name_detector.detect_username(name) == 1:
                    pass
                else:
                    splits = name.split(" ")
                    if len(splits) > 1:
                        answer = input("Is '{}' a user name [Y/n]? ".format(name))
                        logger.info("answer: {}".format(answer))

                        if answer == "n":
                            logger.info("add '{}''s keywords to detector".format(name))
                            non_user_name_detector.update_keywords_data(name)
                        else:
                            # logger.debug
                            logger.info("add new user name '{}'".format(name))
                            non_user_name_detector.update_usernames_data(name)
                    else:
                        logger.info("add '{}''s keywords to detector".format(name))
                        non_user_name_detector.update_keywords_data(name)
                    # raise Exception()

            # raise Exception()
            current_line += 1

    non_user_name_detector.close_file()