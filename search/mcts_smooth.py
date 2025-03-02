import numpy as np
import scipy.special as ssp 


def smooth_heatmap(heatmap: np.ndarray):
    num_nodes = heatmap.shape[-1]
    sorted_vector = np.sort(heatmap, axis=-1)[:, -num_nodes//10].reshape(-1, 1)
    heatmap[(heatmap - sorted_vector) < 0] -= 1e9
    orig_matrix = heatmap
    start = 1.0
    minimum = 0.0
    while minimum < 1e-4: # adjust temperature
        heatmap = ssp.softmax(orig_matrix * start, axis=-1)
        minimum = heatmap[heatmap > 0].min()
        start *= 0.5

    heatmap = np.expand_dims(heatmap, 0)
    return heatmap


def smooth_heatmap_v2(heatmap: np.ndarray, points: np.ndarray):
    assert heatmap.ndim == points.ndim
    if heatmap.ndim == 2:
        num_dim = 2
        heatmap = np.expand_dims(heatmap, axis=0)
        points = np.expand_dims(points, axis=0)
    else:
        num_dim = 3
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    heatmap = heatmap + 0.01 * (1.0 - dists)
    if num_dim == 2:
        heatmap = heatmap[0]
    return heatmap