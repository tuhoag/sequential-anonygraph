import itertools
import logging
import numpy as np

import anonygraph.utils.path as putils
import anonygraph.utils.data as dutils

logger = logging.getLogger(__name__)

class PairDistance(object):
    def __init__(self):
        self.__data = {}
        self.__entity_ids_dict = {}

    def __len__(self):
        return len(self.__data)

    @property
    def num_entities(self):
        return len(self.__entity_ids_dict)

    @property
    def entity_ids(self):
        return list(self.__entity_ids_dict.keys())

    @property
    def entity_ids_set(self):
        return set(self.__entity_ids_dict.keys())

    def get_key(self, entity1_id, entity2_id):
        return (entity1_id, entity2_id)

    def add_entity(self, entity_id):
        if not entity_id in self.__entity_ids_dict:
            entity_idx = len(self.__entity_ids_dict)
            self.__entity_ids_dict[entity_id] = entity_idx

    def add(self, entity1_id, entity2_id, distance):
        key = self.get_key(entity1_id, entity2_id)
        self.__data[key] = distance

        self.add_entity(entity1_id)
        self.add_entity(entity2_id)

    def get_distance(self, entity1_id, entity2_id):
        if entity1_id == entity2_id:
            return 0


        # raise Exception()
        if entity1_id < entity2_id:
            key = self.get_key(entity1_id, entity2_id)
        else:
            key = self.get_key(entity2_id, entity1_id)

        distance = self.__data.get(key, 0)

        # if {entity1_id, entity2_id} == {49, 822}:
        #     logger.debug(self.__data)
        #     logger.debug("key: {} - distance: {}".format(key, distance))
        #     raise Exception()
        # if distance is None:
        #     distance =
        #     raise Exception("entities: {} {} are not existed".format(entity1_id, entity2_id))

        return distance

    def get_distance_matrix(self, entity_ids):
        num_entities = len(entity_ids)
        matrix = np.zeros(shape=(num_entities, num_entities))

        for entity1_idx, entity1_id in enumerate(entity_ids):
            for entity2_idx, entity2_id in enumerate(entity_ids):
                distance = self.get_distance(entity1_id, entity2_id)

                matrix[entity1_idx, entity2_idx] = distance

        return matrix

    def to_distance_matrix(self):
        logger.debug(self.__entity_ids_dict)
        entity_ids = self.entity_ids

        matrix = np.zeros(shape=(self.num_entities, self.num_entities))

        for entity1_id, entity2_id in itertools.product(entity_ids, entity_ids):
            entity1_idx = self.__entity_ids_dict[entity1_id]
            entity2_idx = self.__entity_ids_dict[entity2_id]

            distance = self.get_distance(entity1_id, entity2_id)
            matrix[entity1_idx, entity2_idx] = distance

            # logger.debug("{}, {}, {}".format(entity1_id, entity2_id, distance))

        return matrix

    def get_distance_matrix_of_entity_and_entity(self, entity_ids1, entity_ids2):
        matrix = np.zeros(shape=(len(entity_ids1), len(entity_ids2)))

        for entity1_idx, entity1_id in enumerate(entity_ids1):
            for entity2_idx, entity2_id in enumerate(entity_ids2):
                distance = self.get_distance(entity1_id, entity2_id)

                matrix[entity1_idx, entity2_idx] = distance

            # logger.debug("{}, {}, {}".format(entity1_id, entity2_id, distance))

        return matrix

    def get_entity_id(self, entity_idx):
        entity_ids = self.entity_ids
        return entity_ids[entity_idx]

    @staticmethod
    def from_file(path):
        pair_dist = PairDistance()
        with open(path, 'r') as f:
            for line in f:
                splits = line.strip().split(',')
                entity1_id, entity2_id = map(int, splits[:2])
                distance = float(splits[-1])

                # if {entity1_id,entity2_id} == {49, 822}:
                #     logger.debug("path: {} - line: {} - splits: {} - distance: {}".format(path, line, splits, distance))
                #     raise Exception()
                pair_dist.add(entity1_id, entity2_id, distance)

        return pair_dist


    @staticmethod
    def from_dict(data):
        pair_dist = PairDistance()

        for key, distance in data.items():
            entity1_id, entity2_id = key
            pair_dist.add(entity1_id, entity2_id, distance)

        return pair_dist

    @property
    def max_distance(self):
        return max(self.__data.values())

    @property
    def min_distance(self):
        return min(self.__data.values())
