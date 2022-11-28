import logging
import sys

from anonygraph.info_loss import info

logger = logging.getLogger(__name__)

def test_invalid_signature_size_clusters(clusters, entity2svals, fake_entity_manager, min_signature_size, prefix_str):
    count = 0

    min_size = sys.maxsize
    max_size = -sys.maxsize

    logger.debug("num clusters: {}".format(len(clusters)))
    for cluster in clusters:
        signature = info.get_generalized_signature_info_from_dict(entity2svals, fake_entity_manager, cluster)

        # logger.debug("signature: {} size: {}".format(signature, len(signature)))
        if len(signature) < min_signature_size:
            count += 1

        if len(signature) < min_size:
            min_size = len(signature)

        if len(signature) > max_size:
            max_size = len(signature)

    logger.debug("{} invalid signatures: max size: {} - min size: {}".format(count, max_size, min_size))

    assert count == 0, "[{}] There are {} invalid signature clusters".format(prefix_str, count)


def test_invalid_min_size_clusters(clusters, min_size, prefix_str):
    logger.debug("clusters: {}".format(clusters))
    count = 0

    for cluster in clusters:
        if len(cluster) < min_size:
            logger.debug("cluster {} is invalid (min size: {})".format(cluster, min_size))
            count += 1

    assert count == 0, "[{}] There are {} invalid min size clusters".format(prefix_str, count)

def test_invalid_same_signature_clusters(clusters, entity2svals, fake_entity_manager, historical_table):
    count = 0

    for cluster in clusters:
        signature = info.get_generalized_signature_info_from_dict(entity2svals, fake_entity_manager, cluster)

        for entity_id in cluster:

            previous_signature = historical_table.get_signature(entity_id)

            if signature != previous_signature:
                count += 1

    assert count == 0, "there are {} entities with incorrect signature".format(count)


def test_big_size_clusters(clusters, min_size):
    count = 0

    for cluster in clusters:
        if len(cluster) >= 2 * min_size:
            count += 1

    assert count == 0, "there are {} big size clusters".format(count)

def test_big_signature_clusters(clusters, entity2svals, fake_entity_manager, min_signature_size):
    count = 0

    for cluster in clusters:
        signature = info.get_generalized_signature_info_from_dict(entity2svals, fake_entity_manager, cluster)

        if len(signature) >= 2 * min_signature_size:
            count += 1

    assert count == 0, "there are {} big signatures clusters".format(count)

def test_same_signature_splited_clusters(big_cluster, splited_clusters, entity2svals, fake_entity_manger):
    big_signature = info.get_generalized_signature_info_from_dict(entity2svals, fake_entity_manger, big_cluster)

    count = 0
    for splited_cluster in splited_clusters:
        splited_signature = info.get_generalized_signature_info_from_dict(entity2svals, fake_entity_manger, splited_cluster)

        if splited_signature != big_signature:
            count += 1

    assert count == 0, "There are {} invalid splited signature cluster".format(count)

def test_min_size_deleted_entities(clusters, removed_entity_ids, current_entity_ids, min_size):
    logger.debug("removed_entity_ids: {}".format(removed_entity_ids))
    logger.debug("current_entity_ids: {}".format(current_entity_ids))

    num_entities_in_clusters = sum(map(lambda cluster: len(cluster), clusters))
    entity_ids_in_group = set()
    for cluster in clusters:
        entity_ids_in_group.update(cluster)

    removing_entity_ids = current_entity_ids.difference(entity_ids_in_group)
    logger.debug("removing_entity_ids (len: {}): {}".format(len(removing_entity_ids), removing_entity_ids))

    if len(removed_entity_ids) == 0 or len(removed_entity_ids) >= min_size:
        # expect clusters contain all entities in current_entity_ids
        assert entity_ids_in_group == current_entity_ids, "Clusters ({}) have different entities than those in group ({}): {}".format(clusters, current_entity_ids, removing_entity_ids)
        return

    assert len(removing_entity_ids.intersection(removed_entity_ids)) == 0, "Removing entities ({}) must not be removed_entity_ids ({})".format(removing_entity_ids, removed_entity_ids)

    assert len(removing_entity_ids.union(removed_entity_ids)) >= min_size, "Must have at least min_size ({}) removed and removing entities".format(min_size)


def test_removed_and_removing_entities_not_in_clusters(clusters, removed_entity_ids, current_entity_ids):
    # logger.debug()
    entity_ids_in_group = set()
    for cluster in clusters:
        entity_ids_in_group.update(cluster)

    removing_entity_ids = current_entity_ids.difference(entity_ids_in_group)

    num_invalid_entities = len(removing_entity_ids.union(removed_entity_ids).intersection(entity_ids_in_group))
    assert num_invalid_entities == 0, "removed entities: {} and removing entities: {} must not be in clusters: {} - num_invalid: {}".format(removed_entity_ids, removing_entity_ids, clusters, num_invalid_entities)

def test_min_size_valid_sign2entities(valid_sign2entities, min_size, min_signature_size, history):
    for signature_key, entity_ids_set in valid_sign2entities.items():
        assert len(entity_ids_set) >= min_size, ""