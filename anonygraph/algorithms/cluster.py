import logging

import anonygraph.info_loss.info as info

logger = logging.getLogger(__name__)


class Cluster(object):
    def __init__(self):
        self.__entity_ids = set()

    def add_entity(self, entity_id):
        self.__entity_ids.add(entity_id)

    def to_line_str(self):
        string = ','.join(map(str, self.__entity_ids))
        return string

    def to_list(self):
        return list(self.__entity_ids)

    @property
    def entity_ids(self):
        return self.__entity_ids

    def __iter__(self):
        for entity_id in self.__entity_ids:
            yield entity_id

    @staticmethod
    def from_iter(entity_ids):
        cluster = Cluster()
        cluster.__entity_ids = set(entity_ids)

        return cluster

    def __str__(self):
        return str(self.__entity_ids)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.__entity_ids)

    def has_entity_id(self, entity_id):
        return entity_id in self.__entity_ids