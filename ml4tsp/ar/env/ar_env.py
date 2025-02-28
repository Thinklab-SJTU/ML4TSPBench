import torch
import warnings
import torch.utils.data
from typing import Optional
from torch.utils.data import DataLoader
from torchrl.envs import EnvBase
from torchrl.data import (
    BoundedTensorSpec, CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from ml4co_kit import BaseEnv
from ml4tsp.ar.utils.tensordict import (
    TensorDict, tensordict_collate_fn, 
    load_txt_to_tensordict, TensorDictDataset
)
from ml4tsp.ar.utils.ops import gather_by_index, get_tour_length


class ML4TSPAREnv(BaseEnv, EnvBase):
    batch_locked = False
    def __init__(
        self,
        nodes_num: int,
        min_loc: float = 0,
        max_loc: float = 1,
        random_seed: int = None,
        mode: str = None,
        train_path: str = None,
        val_path: str = None,
        test_path: str = None,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        test_batch_size: int = 1,
        train_data_size: int = 128000,
        val_data_size: int = 1280,
        test_data_size: int = 1280,
        num_workers: int = 1,
        torchrl_mode: bool = False,
        device: str = "cpu",
    ):
        super(ML4TSPAREnv, self).__init__(
            name="tsp_ar",
            mode=mode,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers
        )
        super(BaseEnv, self).__init__(device=device, batch_size=[])
        self.nodes_num = nodes_num
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.train_data_size = train_data_size
        self.val_data_size = val_data_size
        self.test_data_size = test_data_size
        self._make_spec(td_params=None)
        
        # torchrl
        self._torchrl_mode = torchrl_mode
        
        # random seed
        if random_seed is None:
            random_seed = torch.empty((), dtype=torch.int64).random_().item()
        self._set_seed(random_seed)

        # load data
        self.load_data()
        
    def _set_seed(self, seed: Optional[int]):
        """Set the seed for the environment"""
        rng = torch.manual_seed(seed)
        self.rng = rng

    def generate_data(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        locs = (
            torch.rand((*batch_size, self.nodes_num, 2), generator=self.rng)
            * (self.max_loc - self.min_loc)
            + self.min_loc
        )
        return TensorDict({"locs": locs}, batch_size=batch_size, device=self.device)
        
    def load_data(self):
        if self.mode == "train":
            # train dataset
            if self.train_path is not None:
                message = (
                    "Loading training dataset from file. This may not be desired in RL since "
                    "the dataset is fixed and the agent will not be able to explore new states"
                )
                warnings.warn(message)
                self.train_dataset, self.train_data_size = load_txt_to_tensordict(self.train_path)
            else:
                self.train_dataset = TensorDictDataset(self.generate_data(self.train_data_size))

            # val dataset
            if self.val_path is not None:
                self.val_dataset, self.val_data_size = load_txt_to_tensordict(self.val_path)
            else:
                self.val_dataset = TensorDictDataset(self.generate_data(self.val_data_size))

        elif self.mode == "test":
            # test dataset
            if self.test_path is not None:
                self.test_dataset, self.test_data_size = load_txt_to_tensordict(self.test_path)
            else:
                self.test_dataset = TensorDictDataset(self.generate_data(self.test_data_size))
        
        else:
            # solve mode / none mode
            pass
    
    def _dataloader(self, dataset, batch_size: int, shuffle: bool = False):
        data = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=tensordict_collate_fn,
        )
        return data
    
    def train_dataloader(self):
        return self._dataloader(self.train_dataset, self.train_batch_size, True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, self.val_batch_size, False)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, self.test_batch_size, False)
    
    def step(self, td: TensorDict) -> TensorDict:
        """Step function to call at each step of the episode containing an action.
        If `_torchrl_mode` is True, we call `_torchrl_step` instead which set the
        `next` key of the TensorDict to the next state - this is the usual way to do it in TorchRL,
        but inefficient in our case
        """
        if not self._torchrl_mode:
            # Default: just return the TensorDict without farther checks etc is faster
            td = self._step(td)
            return {"next": td}
        else:
            # Since we simplify the syntax
            return self._torchrl_step(td)
    
    def _torchrl_step(self, td: TensorDict) -> TensorDict:
        """See :meth:`super().step` for more details.
        This is the usual way to do it in TorchRL, but inefficient in our case

        Note:
            Here we clone the TensorDict to avoid recursion error, since we allow
            for directly updating the TensorDict in the step function
        """
        # sanity check
        self._assert_tensordict_shape(td)
        next_preset = td.get("next", None)

        next_tensordict = self._step(
            td.clone()
        )  # NOTE: we clone to avoid recursion error
        next_tensordict = self._step_proc_data(next_tensordict)
        if next_preset is not None:
            next_tensordict.update(next_preset.exclude(*next_tensordict.keys(True, True)))
        td.set("next", next_tensordict)
        return td
    
    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_node = td["action"]
        first_node = current_node if td["i"].all() == 0 else td["first_node"]

        # # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )

        # We are done there are no unvisited locations
        done = torch.sum(available, dim=-1) == 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": available,
                "reward": reward,
                "done": done,
            },
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialize locations
        init_locs = td["locs"] if td is not None else None
        if batch_size is None:
            batch_size = self.batch_size if init_locs is None else init_locs.shape[:-2]
        device = init_locs.device if init_locs is not None else self.device
        self.to(device)
        if init_locs is None:
            init_locs = self.generate_data(batch_size=batch_size).to(device)["locs"]
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # We do not enforce loading from self for flexibility
        num_loc = init_locs.shape[-2]

        # Other variables
        current_node = torch.zeros((batch_size), dtype=torch.int64, device=device)
        available = torch.ones(
            (*batch_size, num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "locs": init_locs,
                "first_node": current_node,
                "current_node": current_node,
                "i": i,
                "action_mask": available,
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
            },
            batch_size=batch_size,
        )

    def _make_spec(self, td_params):
        """Make the observation and action specs from the parameters"""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(self.nodes_num, 2),
                dtype=torch.float32,
            ),
            first_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.nodes_num),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.input_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            minimum=0,
            maximum=self.nodes_num,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    @staticmethod
    def get_reward(td, actions) -> TensorDict:
        locs = td["locs"]
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather locations in order of tour and return distance between them (i.e., -reward)
        locs_ordered = gather_by_index(locs, actions)
        return -get_tour_length(locs_ordered)

        
def batch_to_scalar(param):
    """Return first element if in batch. Used for batched parameters that are the same for all elements in the batch."""
    if len(param.shape) > 0:
        return param[0].item()
    if isinstance(param, torch.Tensor):
        return param.item()
    return param