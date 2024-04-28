def calc_indim(features: dict):
    in_dim = 0
    for v in features.values():
        in_dim += v[0]
    return in_dim


def calc_in_channels(features: dict):
    return len(features)
