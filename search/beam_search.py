from multiprocessing import Pool
import numpy as np
import torch
from scipy.spatial.distance import cdist
import scipy.sparse
import scipy.spatial


class Beamsearch(object):
    """
    Class for managing internals of beamsearch procedure.

    References:
        General: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam.py
        For TSP: https://github.com/alexnowakvila/QAP_pt/blob/master/src/tsp/beam_search.py
    """
    def __init__(self, beam_size, batch_size, num_nodes,
                 dtypeFloat=torch.FloatTensor, dtypeLong=torch.LongTensor, 
                 probs_type='raw', random_start=False):
        """
        Args:
            beam_size: Beam size
            batch_size: Batch size
            num_nodes: Number of nodes in TSP tours
            dtypeFloat: Float data type (for GPU/CPU compatibility)
            dtypeLong: Long data type (for GPU/CPU compatibility)
            probs_type: Type of probability values being handled by beamsearch (either 'raw'/'logits'/'argmax'(TODO))
            random_start: Flag for using fixed (at node 0) vs. random starting points for beamsearch
        """
        # Beamsearch parameters
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.num_nodes = num_nodes
        self.probs_type = probs_type
        # Set data types
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        # Set beamsearch starting nodes
        self.start_nodes = torch.zeros(batch_size, beam_size).type(self.dtypeLong)
        if random_start == True:
            # Random starting nodes
            self.start_nodes = torch.randint(0, num_nodes, (batch_size, beam_size)).type(self.dtypeLong)
        # Mask for constructing valid hypothesis
        self.mask = torch.ones(batch_size, beam_size, num_nodes).type(self.dtypeFloat)
        self.update_mask(self.start_nodes)  # Mask the starting node of the beam search
        # Score for each translation on the beam
        self.scores = torch.zeros(batch_size, beam_size).type(self.dtypeFloat)
        self.all_scores = []
        # Backpointers at each time-step
        self.prev_Ks = []
        # Outputs at each time-step
        self.next_nodes = [self.start_nodes]

    def get_current_state(self):
        """
        Get the output of the beam at the current timestep.
        """
        current_state = (self.next_nodes[-1].unsqueeze(2)
                         .expand(self.batch_size, self.beam_size, self.num_nodes))
        return current_state

    def get_current_origin(self):
        """
        Get the backpointers for the current timestep.
        """
        return self.prev_Ks[-1]

    def advance(self, trans_probs):
        """
        Advances the beam based on transition probabilities.

        Args:
            trans_probs: Probabilities of advancing from the previous step (batch_size, beam_size, num_nodes)
        """

        # Compound the previous scores (summing logits == multiplying probabilities)
        if len(self.prev_Ks) > 0:
            if self.probs_type == 'raw':
                beam_lk = trans_probs * self.scores.unsqueeze(2).expand_as(trans_probs)
            elif self.probs_type == 'logits':
                beam_lk = trans_probs + self.scores.unsqueeze(2).expand_as(trans_probs)
        else:
            beam_lk = trans_probs
            # Only use the starting nodes from the beam
            if self.probs_type == 'raw':
                beam_lk[:, 1:] = torch.zeros(beam_lk[:, 1:].size()).type(self.dtypeFloat)
            elif self.probs_type == 'logits':
                beam_lk[:, 1:] = -torch.inf * torch.ones(beam_lk[:, 1:].size()).type(self.dtypeFloat)

        # Multiply by mask
        beam_lk = beam_lk * self.mask
        beam_lk = beam_lk.view(self.batch_size, -1)  # (batch_size, beam_size * num_nodes)
        
        # Get top k scores and indexes (k = beam_size)
        bestScores, bestScoresId = beam_lk.topk(self.beam_size, 1, True, True)

        # Update scores
        self.scores = bestScores
            
        # Update backpointers
        prev_k = bestScoresId // self.num_nodes
        self.prev_Ks.append(prev_k)

        # Update outputs
        new_nodes = bestScoresId - prev_k * self.num_nodes
        self.next_nodes.append(new_nodes)

        # Re-index mask
        perm_mask = prev_k.unsqueeze(2).expand_as(self.mask)  # (batch_size, beam_size, num_nodes)

        self.mask = self.mask.gather(1, perm_mask)
        # Mask newly added nodes
        self.update_mask(new_nodes)

    def update_mask(self, new_nodes):
        """
        Sets new_nodes to zero in mask.
        """
        arr = (torch.arange(0, self.num_nodes).unsqueeze(0).unsqueeze(1)
               .expand_as(self.mask).type(self.dtypeLong))
        new_nodes = new_nodes.unsqueeze(2).expand_as(self.mask)
        update_mask = 1 - torch.eq(arr, new_nodes).type(self.dtypeFloat)
        if self.probs_type == 'logits':
            # Convert 0s in mask to inf
            update_mask[update_mask == 0] = torch.inf
        self.mask = self.mask * update_mask

    def sort_best(self):
        """
        Sort the beam.
        """
        return torch.sort(self.scores, 0, True)

    def get_best(self):
        """
        Get the score and index of the best hypothesis in the beam.
        """
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    def get_hypothesis(self, k):
        """
        Walk back to construct the full hypothesis.

        Args:
            k: Position in the beam to construct (usually 0s for most probable hypothesis)
        """
        assert self.num_nodes == len(self.prev_Ks) + 1

        hyp = -1 * torch.ones(self.batch_size, self.num_nodes).type(self.dtypeLong)
        for j in range(len(self.prev_Ks) - 1, -2, -1):
            hyp[:, j + 1] = self.next_nodes[j + 1].gather(1, k).view(1, self.batch_size)
            k = self.prev_Ks[j].gather(1, k)
        return hyp


def tour_nodes_to_tour_len(nodes, W_values):
    """
    Helper function to calculate tour length from ordered list of tour nodes.
    """
    tour_len = 0
    for idx in range(len(nodes) - 1):
        i = nodes[idx]
        j = nodes[idx + 1]
        tour_len += W_values[i][j]
    # Add final connection of tour in edge target
    tour_len += W_values[j][nodes[0]]
    return tour_len
  

def is_valid_tour(nodes, num_nodes):
    """
    Sanity check: tour visits all nodes given.
    """
    return sorted(nodes) == [i for i in range(num_nodes)]
  

def _tsp_beam(adj_mat: np.ndarray, np_points: np.ndarray, kwargs):
    '''
    Output: tours
    Reference: https://github.com/chaitjo/graph-convnet-tsp
    '''
    if adj_mat.ndim == 2:
        adj_mat = np.expand_dims(adj_mat, axis=0)
        batch_size, num_nodes, _ = adj_mat.shape
        x_edges_values = np.expand_dims(cdist(np_points, np_points), axis=0)
    else:
        batch_size, num_nodes, _ = adj_mat.shape
        x_edges_values = np.array([cdist(coords, coords) for coords in np_points])

    beam_size = kwargs['beam_size']
    probs_type = kwargs['beam_probs_type']
    beamsearch = Beamsearch(
        beam_size=beam_size, 
        batch_size=batch_size, 
        num_nodes=num_nodes,
        probs_type=probs_type, 
        random_start=kwargs['beam_random_smart']
    )
    
    adj_mat = torch.clamp(torch.tensor(adj_mat), 1e-14, 1-1e-6)

    if probs_type == 'logits':
        adj_mat = torch.log(adj_mat)
    trans_probs = adj_mat.gather(1, beamsearch.get_current_state())

    for _ in range(num_nodes - 1):
        beamsearch.advance(trans_probs)
        trans_probs = adj_mat.gather(1, beamsearch.get_current_state())

    ends = torch.zeros(batch_size, 1).long()
    shortest_tours = beamsearch.get_hypothesis(ends)
    shortest_lens = [1e6] * len(shortest_tours)

    for idx in range(len(shortest_tours)):
        shortest_lens[idx] = tour_nodes_to_tour_len(shortest_tours[idx].cpu().numpy(),
                                                    x_edges_values[idx])
        
    for pos in range(1, beam_size):
        ends = pos * torch.ones(batch_size, 1).long() # New positions
        hyp_tours = beamsearch.get_hypothesis(ends)
        for idx in range(len(hyp_tours)):
            hyp_nodes = hyp_tours[idx].cpu().numpy()
            hyp_len = tour_nodes_to_tour_len(hyp_nodes, x_edges_values[idx])
            # Replace tour in shortest_tours if new length is shorter than current best
            if hyp_len < shortest_lens[idx] and is_valid_tour(hyp_nodes, num_nodes):
                shortest_tours[idx] = hyp_tours[idx]
                shortest_lens[idx] = hyp_len

    shortest_tours = torch.cat([shortest_tours, 
            shortest_tours[:, 0].reshape(batch_size, 1)], dim=1)    
    if batch_size == 1:
        shortest_tours = shortest_tours.squeeze()
    shortest_tours = shortest_tours.tolist()
    return shortest_tours


def tsp_beam(adj_mat: np.ndarray, np_points: np.ndarray, edge_index_np: np.ndarray=None, 
             sparse_graph=False, parallel_sampling=1, device="cpu", **kwargs):
    """_summary_

    Args:
        adj_mat (np.ndarray): the predict heatmap (B, N, N)
        np_points (np.ndarray): the coords of nodes (B, N, 2)
        edge_index_np (np.ndarray, optional): the edge_index for sparse heatmap. Defaults to None.
        sparse_graph (bool, optional): whether the graph is sparse. Defaults to False.
        parallel_sampling (int, optional): _description_. Defaults to 1.
        device (str, optional): _description_. Defaults to "cpu".
        args (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)
    if not sparse_graph:
        splitted_adj_mat = [
            (adj_mat[0] + adj_mat[0].T)/2 for adj_mat in splitted_adj_mat
        ]
    else:
        splitted_adj_mat = [
            scipy.sparse.coo_matrix(
                ((adj_mat+1e-14)/2, (edge_index_np[0], edge_index_np[1])),
            ).toarray() + scipy.sparse.coo_matrix(
                ((adj_mat+1e-14)/2, (edge_index_np[1], edge_index_np[0])),
            ).toarray() +1e-14 for adj_mat in splitted_adj_mat
        ]

    splitted_points = [np_points for _ in range(parallel_sampling)]
    spliited_kwargs = [kwargs for _ in range(parallel_sampling)]
    
    if np_points.shape[0] > 1000 and parallel_sampling > 1:
        with Pool(parallel_sampling) as p:
            results = p.starmap(
                _tsp_beam,
                zip(splitted_adj_mat, splitted_points, spliited_kwargs),
            )
    else:
        results = [
            _tsp_beam(_adj_mat, _np_points, _args) for _adj_mat, _np_points, _args \
                in zip(splitted_adj_mat, splitted_points, spliited_kwargs)
        ]

    return results
