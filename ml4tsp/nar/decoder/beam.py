import torch
import numpy as np
from ml4co_kit import to_numpy, to_tensor, TSPEvaluator
from ml4tsp.nar.decoder.base import ML4TSPNARDecoder


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


class ML4TSPNARBeamDecoder(ML4TSPNARDecoder):
    def __init__(
        self,
        heatmap_delta: float = 1e-14,
        active_search: bool = False,
        as_steps: int = 100,
        as_samples: int = 1000,
        as_inner_lr: float = 5e-2,
        beam_size: int = 1280,
        probs_type: str = "logits",
        random_start: bool = False,
        return_best: bool = True
    ):
        super(ML4TSPNARBeamDecoder, self).__init__(
            heatmap_delta=heatmap_delta, active_search=active_search, 
            as_steps=as_steps, as_samples=as_samples, as_inner_lr=as_inner_lr
        )
        
        # beamsearch parameters
        self.beam_size = beam_size
        self.probs_type = probs_type
        self.random_start = random_start
        self.return_best = return_best
        
        # set data types
        self.dtypeFloat = torch.FloatTensor
        self.dtypeLong = torch.LongTensor        
        
    def beam_init(self):
        # set beamsearch starting nodes
        if not self.random_start:
            self.start_nodes = torch.zeros(self.batch_size, self.beam_size).type(self.dtypeLong)
        else:
            self.start_nodes = torch.randint(
                low=0, high=self.nodes_num, 
                size=(self.batch_size, self.beam_size)
            ).type(self.dtypeLong)
        
        # mask for constructing valid hypothesis
        self.mask = torch.ones(
            self.batch_size, self.beam_size, self.nodes_num
        ).type(self.dtypeFloat)
        
        # mask the starting node of the beam search
        self.update_mask(self.start_nodes)
        
        # score for each translation on the beam
        self.scores = torch.zeros(self.batch_size, self.beam_size).type(self.dtypeFloat)
        self.all_scores = []
        
        # backpointers at each time-step
        self.prev_Ks = []
        
        # outputs at each time-step
        self.next_nodes = [self.start_nodes]

    def update_mask(self, new_nodes: torch.Tensor):
        """
        Sets new_nodes to zero in mask.
        """
        array = torch.arange(0, self.nodes_num).unsqueeze(dim=0).unsqueeze(dim=1)
        array = array.expand_as(self.mask).type(self.dtypeLong)
        new_nodes = new_nodes.unsqueeze(dim=2).expand_as(self.mask)
        update_mask = 1 - torch.eq(array, new_nodes).type(self.dtypeFloat)
        if self.probs_type == 'logits':
            update_mask[update_mask == 0] = torch.inf
        self.mask = self.mask * update_mask

    def get_current_state(self):
        """
        Get the output of the beam at the current timestep.
        """
        last_nodes = self.next_nodes[-1]
        last_nodes: torch.Tensor
        current_state = last_nodes.unsqueeze(2)
        current_state = (current_state.expand(self.batch_size, self.beam_size, self.nodes_num))
        return current_state
    
    def advance(self, trans_probs: torch.Tensor):
        # compound the previous scores (summing logits == multiplying probabilities)
        self.scores: torch.Tensor
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

        # multiply by mask
        beam_lk: torch.Tensor
        beam_lk = beam_lk * self.mask
        beam_lk = beam_lk.view(self.batch_size, -1)  # (batch_size, beam_size * num_nodes)
        
        # get top k scores and indexes (k = beam_size)
        bestScores, bestScoresId = beam_lk.topk(self.beam_size, 1, True, True)

        # update scores
        self.scores = bestScores
            
        # update backpointers
        prev_k = bestScoresId // self.nodes_num
        prev_k: torch.Tensor
        self.prev_Ks.append(prev_k)

        # update outputs
        new_nodes = bestScoresId - prev_k * self.nodes_num
        self.next_nodes.append(new_nodes)

        # re-index mask
        perm_mask = prev_k.unsqueeze(2).expand_as(self.mask)  # (batch_size, beam_size, num_nodes)
        self.mask: torch.Tensor
        self.mask = self.mask.gather(1, perm_mask)
        
        # mask newly added nodes
        self.update_mask(new_nodes)
        
    def get_hypothesis(self, k):
        """
        Walk back to construct the full hypothesis.

        Args:
            k: Position in the beam to construct (usually 0s for most probable hypothesis)
        """
        assert self.nodes_num == len(self.prev_Ks) + 1
        hyp = -1 * torch.ones(self.batch_size, self.nodes_num).type(self.dtypeLong)
        for j in range(len(self.prev_Ks) - 1, -2, -1):
            hyp[:, j + 1] = self.next_nodes[j + 1].gather(1, k).view(1, self.batch_size)
            k = self.prev_Ks[j].gather(1, k)
        return hyp
    
    def _decode(self, heatmap: np.ndarray, points: np.ndarray) -> np.ndarray:
        # np.ndarray -> torch.Tensor
        heatmap = to_tensor(heatmap)
        points = to_tensor(points)
        
        # beam init
        self.batch_size, _, self.nodes_num = heatmap.shape
        self.beam_init()
        
        # if probs_type is logits, heatmap requires torch.log processing 
        if self.probs_type == "logits":
            heatmap = torch.log(heatmap) - self.heatmap_delta

        # beam search
        heatmap: torch.Tensor
        trans_probs = heatmap.gather(1, self.get_current_state())
        for _ in range(self.nodes_num - 1):
            self.advance(trans_probs)
            trans_probs = heatmap.gather(1, self.get_current_state())
        
        # gain the tours
        tours = list()
        for idx in range(self.beam_size):
            ends = idx * torch.ones(self.batch_size, 1).long()
            _tours = to_numpy(self.get_hypothesis(ends))
            tours.append(_tours)
        tours = np.array(tours).transpose(1, 0, 2)
        zeros = np.zeros(shape=(self.batch_size, self.beam_size, 1))
        tours = np.concatenate([tours, zeros], axis=-1).astype(np.int64)
        
        # if return_best is True, return the best tour
        # else, return all of the valid tours
        points = to_numpy(points)
        if self.return_best:
            best_tours = list()
            for idx, _tours in enumerate(tours):
                eval = TSPEvaluator(points[idx])
                best_tour = None
                best_cost = None
                for tour in _tours:
                    if self.is_valid_tour(tour):
                        cost = eval.evaluate(tour)
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best_tour = tour
                if best_tour is None:
                    raise ValueError()
                best_tours.append(best_tour)
            tours = np.expand_dims(np.array(best_tours), axis=1)
        return tours