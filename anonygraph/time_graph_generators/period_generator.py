import logging
import math

logger = logging.getLogger(__name__)

def _convert_time(value, src_unit, dest_unit):
    if src_unit == dest_unit:
        return value

    if src_unit == 'second' and dest_unit == 'day':
        result = value / 3600 / 24
    elif src_unit == 'second' and dest_unit == 'week':
        result = value / 3600 / 24 / 7
    else:
        raise NotImplementedError("Cannot convert from {} to {}".format(src_unit, dest_unit))

    return result

def _test_group_time_instances(groups, time_instances):
    num_instances = 0
    for day, group in groups.items():
        # logger.debug('day: {} - num times: {}'.format(day, len(group)))
        num_instances += len(group)

    assert num_instances == len(time_instances)

class PeriodGenerator(object):
    def __init__(self, period, dest_unit='week'):
        self.period = period
        self.dest_unit = dest_unit

    def __group_time_instances(self, time_instances, unit):
        min_time = time_instances[0]

        groups = {}

        for t in time_instances:
            relative_time = _convert_time(t - min_time, unit, self.dest_unit)
            group_id = math.floor(relative_time / self.period)

            group = groups.get(group_id)
            if group is None:
                group = []
                groups[group_id] = group

            group.append(t)

        return groups


    def run(self, graph):
        unit = graph.time_unit

        time_instances = sorted(graph.time_instances)
        logger.debug("from {} to {} {}".format(time_instances[0], time_instances[-1], unit))

        # group set of periods
        time_groups = self.__group_time_instances(time_instances, unit)

        logger.debug("number of groups: {}: {}".format(len(time_groups), time_groups.keys()))
        _test_group_time_instances(time_groups, time_instances)

        logger.debug(list(time_groups.keys()))

        return time_groups