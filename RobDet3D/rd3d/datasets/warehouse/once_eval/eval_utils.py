import numpy as np


def compute_split_parts(num_samples, num_parts):
    part_samples = num_samples // num_parts
    remain_samples = num_samples % num_parts
    if part_samples == 0:
        return [num_samples]
    if remain_samples == 0:
        return [part_samples] * num_parts
    else:
        return [part_samples] * num_parts + [remain_samples]


def overall_filter(boxes):
    ignore = np.zeros(boxes.shape[0], dtype=bool)  # all false
    return ignore


def distance_filter(boxes, level):
    ignore = np.ones(boxes.shape[0], dtype=bool)  # all true
    dist = np.sqrt(np.sum(boxes[:, 0:3] * boxes[:, 0:3], axis=1))

    if level == 0:  # 0-10m
        flag = dist < 10
    elif level == 1:  # 10-20m
        flag = (dist >= 10) & (dist < 20)
    elif level == 2:  # 20m-inf
        flag = dist >= 20
    else:
        assert False, 'level < 3 for distance metric, found level %s' % (str(level))

    ignore[flag] = False
    return ignore


def overall_distance_filter(boxes, level):
    ignore = np.ones(boxes.shape[0], dtype=bool)  # all true
    dist = np.sqrt(np.sum(boxes[:, 0:3] * boxes[:, 0:3], axis=1))

    if level == 0:
        flag = np.ones(boxes.shape[0], dtype=bool)
    elif level == 1:  # 0-10m
        flag = dist < 10
    elif level == 2:  # 10-20m
        flag = (dist >= 10) & (dist < 20)
    elif level == 3:  # 20m-inf
        flag = dist >= 20
    else:
        assert False, 'level < 4 for overall & distance metric, found level %s' % (str(level))

    ignore[flag] = False
    return ignore