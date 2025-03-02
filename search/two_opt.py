import numpy as np
import torch


def batched_two_opt_torch(
    points: np.ndarray, 
    tour: np.ndarray, 
    max_iterations: int=5000, 
    device="cpu"
):
    iterator = 0
    tour = tour.copy()
    if tour.ndim == 1:
        tour = np.expand_dims(tour, axis=0)
    with torch.inference_mode():
        cuda_points = torch.from_numpy(points).to(device)
        cuda_tour = torch.from_numpy(tour).to(device)
        batch_size = cuda_tour.shape[0]
        min_change = -1.0

        while min_change < 0.0:
            points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
            points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
            points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
            points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))
            
            A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
            A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
            A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
            A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

            change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
            valid_change = torch.triu(change, diagonal=2)

            min_change = torch.min(valid_change)
            flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
            min_i = torch.div(flatten_argmin_index, len(points), rounding_mode='floor')
            min_j = torch.remainder(flatten_argmin_index, len(points))

            if min_change < -1e-6:
                for i in range(batch_size):
                    cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
                iterator += 1
            else:
                break

            if iterator >= max_iterations:
                break
        tour = cuda_tour.cpu().numpy()

    return tour, iterator


def tsp_2opt(
    np_points: np.ndarray, 
    tours: np.ndarray, 
    adj_mat: np.ndarray=None, 
    device="cpu",
    **kwargs
):
    '''
    Output: tours, shape (parallel_sampling, N + 1)
    '''
    tours = np.array(tours)
    max_iterations_2opt = kwargs.get("max_iterations_2opt", 5000)
    solved_tours, _ = batched_two_opt_torch(
        np_points.astype("float64"), 
        tours.astype('int64'),
        max_iterations=max_iterations_2opt, 
        device=device
    )
    if tours.ndim == 1:
        solved_tours = solved_tours[0]
    return solved_tours