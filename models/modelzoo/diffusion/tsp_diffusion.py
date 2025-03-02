import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from tqdm import tqdm
from typing import Any, Union
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data as GraphData
from torch_sparse import SparseTensor

from models.utils.evaluator import TSPEvaluator
from models.utils.active_search import ActiveSearch
from utils.utils import coords_to_distance
from search import tsp_greedy, get_decoding_func, get_local_search_func
from .diffusion_base import MetaDiffusion
import pygmtools as pygm

from models.utils.diffusion import get_schedule_fn, CategoricalDiffusion, InferenceSchedule


tsp_diffusion_path = {
    50:  'https://huggingface.co/ML4TSPBench/DIFFUSION/resolve/main/tsp50_diffusion.pt?download=true',
    100: 'https://huggingface.co/ML4TSPBench/DIFFUSION/resolve/main/tsp100_diffusion.pt?download=true',
    500: 'https://huggingface.co/ML4TSPBench/DIFFUSION/resolve/main/tsp500_diffusion.pt?download=true'
}


class TSPDiffusion(MetaDiffusion):
    def __init__(
        self,
        num_nodes: int=50,
        # network
        network_type: str="gnn",
        input_dim: int=2,
        embedding_dim: int=128,
        hidden_dim: int=256,
        output_channels: int=2,
        num_layers: int=12,
        sparse_factor: int=-1, 
        # train/valid/test
        train_batch_size: int=64,
        valid_batch_size: int=1,
        test_batch_size: int=1,
        train_file: str=None,
        valid_file: str=None,
        test_file: str=None,
        valid_samples: int=1280,
        mode: str=None,
        num_workers: int=0,
        # parallel/active
        parallel_sampling: int=1,
        active_search: bool=False,
        as_steps: int=500,
        as_samples: int=1000,
        gradient_search: bool=False,
        rewrite_steps: int=3,
        rewrite_ratio: float=0.25,
        steps_inf: int=10,
        # test_step
        decoding_type: str="greedy",
        local_search_type: str=None,
        **kwargs
    ):
        super(TSPDiffusion, self).__init__(
            num_nodes=num_nodes,
            network_type=network_type,
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_channels=output_channels,
            num_layers=num_layers,
            sparse_factor=sparse_factor,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            test_batch_size=test_batch_size,
            train_file=train_file,
            valid_file=valid_file,
            test_file=test_file,
            valid_samples=valid_samples,
            mode=mode,
            num_workers=num_workers,
            **kwargs
        )
        self.parallel_sampling = parallel_sampling
        self.active_search = active_search
        self.as_steps = as_steps
        self.as_samples = as_samples
        self.gradient_search = gradient_search
        self.rewrite_steps = rewrite_steps
        self.rewrite_ratio = rewrite_ratio
        self.steps_inf = steps_inf

        self.test_decoding_type = decoding_type
        self.test_decoding_kwargs= kwargs
        self.test_ls_type = local_search_type
        self.test_ls_kwargs = kwargs

        self.output_channels = output_channels

    def forward(self, x, adj, t, edge_index):
        h = self.model(x, graph=adj, edge_index=edge_index, timesteps=t)
        return h # .view(-1, self.output_channels, self.num_nodes, self.num_nodes)

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        edge_index = None
        original_edge_index = None
        np_edge_index = None
        device = batch[-1].device
        if not self.sparse:
            real_batch_idx, points, adj_matrix, gt_tour = batch
            points: torch.Tensor
            gt_tour: torch.Tensor
            # deal with different mode
            if phase == 'test':
                points = points.unsqueeze(dim=0)
                gt_tour = gt_tour.unsqueeze(dim=0)
                adj_matrix = adj_matrix.unsqueeze(dim=0)
            np_points = points.cpu().numpy()[0]
            np_gt_tour = gt_tour.cpu().numpy()[0]
            t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)
        else:
            real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch[0:5]
            t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
            route_edge_flags = graph_data.edge_attr
            points = graph_data.x
            edge_index = graph_data.edge_index
            num_edges = edge_index.shape[1]
            batch_size = point_indicator.shape[0]
            adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
        
        # Sample from diffusion
        adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
        if self.sparse:
            adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1)

        xt = self.diffusion.sample(adj_matrix_onehot, t)
        xt = xt * 2 - 1
        xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

        if self.sparse:
            t = torch.from_numpy(t).float()
            t = t.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
            xt = xt.reshape(-1)
            adj_matrix = adj_matrix.reshape(-1)
            points = points.reshape(-1, 2)
            edge_index = edge_index.to(adj_matrix.device).reshape(2, -1)
            original_edge_index = edge_index.clone().cpu()
            np_points = points.cpu().numpy()
            np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
            np_edge_index = edge_index.cpu().numpy()
        else:
            t = torch.from_numpy(t).float().view(adj_matrix.shape[0])


        # Denoise
        # points = points.unsqueeze(dim=0) if points.ndim == 2 else points
        # xt = xt.unsqueeze(dim=0) if xt.ndim == 2 else xt
        x0_pred = self.forward(
            points.float().to(adj_matrix.device),
            xt.float().to(adj_matrix.device),
            t.float().to(adj_matrix.device),
            edge_index,
        )

        # Compute loss
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(x0_pred, adj_matrix.long())

        # return loss if current is a training step
        if phase == "train":
            metrics = {"train/loss": loss}
            for k, v in metrics.items():
                self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
            return loss
        
        # Solve
        stacked_tours = []
      
        if self.parallel_sampling > 1:
            if not self.sparse:
                points = points.repeat(self.parallel_sampling, 1, 1)
            else:
                points = points.repeat(self.parallel_sampling, 1)
                edge_index = self.duplicate_edge_index(edge_index, np_points.shape[0], device)

        xt = torch.randn_like(adj_matrix.float())
        if self.parallel_sampling > 1:
            if not self.sparse:
                xt = xt.repeat(self.parallel_sampling, 1, 1)
            else:
                xt = xt.repeat(self.parallel_sampling, 1)  # [B, E]
            xt = torch.randn_like(xt)
        xt = (xt > 0).long()

        if self.sparse:
            xt = xt.reshape(-1)

        steps = self.inference_diffusion_steps
        time_schedule = InferenceSchedule(inference_schedule=self.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)
        
        # Diffusion iterations
        for i in range(steps):
            t1, t2 = time_schedule(i)
            t1 = np.array([t1]).astype(int)
            t2 = np.array([t2]).astype(int)

            # [B, N, N], heatmap score
            xt = self.categorical_denoise_step(
                points, xt, t1, device, edge_index, target_t=t2)

        adj_mat = xt
        
        # Decoding / solve
        if phase == "val":
            # Heatmap 
            adj_mat = adj_mat.float().cpu().detach().numpy() + 1e-6
            solved_tours = tsp_greedy(
                adj_mat=adj_mat, 
                np_points=np_points, 
                edge_index_np=np_edge_index, 
                sparse_graph=self.sparse, 
                parallel_sampling=self.parallel_sampling,
                device=gt_tour.device,
            )
        else:   
            # Active search
            if self.active_search:
                dist_mat = torch.cdist(points, points)
                AS = ActiveSearch(dist_mat, self.as_steps, self.as_samples)
                adj_mat = AS.active_search(adj_mat.clone().detach())
            adj_mat = adj_mat.float().cpu().detach().numpy() + 1e-6

            # decode
            decoding_func = get_decoding_func(task="tsp", name=self.test_decoding_type)
            solved_tours = decoding_func(
                adj_mat=adj_mat, 
                np_points=np_points, 
                edge_index_np=np_edge_index, 
                sparse_graph=self.sparse, 
                device=gt_tour.device,
                parallel_sampling=self.parallel_sampling,
                **self.test_decoding_kwargs
            )

            # local_search
            local_search_func = get_local_search_func(task="tsp", name=self.test_ls_type)
            # if local_search_func is not None:
            #     solved_tours = local_search_func(
            #         np_points=np_points, 
            #         tours=solved_tours, 
            #         adj_mat=adj_mat, 
            #         device=gt_tour.device,
            #         **self.test_ls_kwargs
            #     )

            if local_search_func is not None:
                if self.sparse:
                    if self.parallel_sampling == 1:
                        sparse_adj_mat = SparseTensor(
                            row=edge_index[0],
                            col=edge_index[1],
                            value=torch.tensor(adj_mat).to(device=edge_index.device)
                        )
                        adj_mat = sparse_adj_mat.to_dense().unsqueeze(dim=0).cpu().numpy()
                    else:
                        sparse_adj_mat = list()
                        ps_edge_index = edge_index.reshape(2, self.parallel_sampling, -1).transpose(0, 1)
                        ps_adj_mat = adj_mat.reshape(self.parallel_sampling, -1)
                        for idx in range(self.parallel_sampling):
                            ps_sparse_adj_mat = SparseTensor(
                                row=ps_edge_index[idx][0] - self.num_nodes*idx,
                                col=ps_edge_index[idx][1] - self.num_nodes*idx,
                                value=torch.tensor(ps_adj_mat[idx]).to(device=edge_index.device)
                            )
                            sparse_adj_mat.append(ps_sparse_adj_mat.to_dense().unsqueeze(dim=0).cpu().numpy()[0])
                        adj_mat = np.array(sparse_adj_mat)

                solved_tours = local_search_func(
                    np_points=np_points, 
                    tours=solved_tours, 
                    adj_mat=adj_mat, 
                    device=gt_tour.device,
                    **self.test_ls_kwargs
                )
            
        # Check the tours
        for idx in range(len(solved_tours)):
            assert sorted(solved_tours[idx][:-1]) == [i for i in range(self.num_nodes)]

        stacked_tours.append(solved_tours)
        solved_tours = np.concatenate(stacked_tours, axis=0)

        # Caculate the gap
        tsp_solver = TSPEvaluator(np_points)
        gt_cost = tsp_solver.evaluate(np_gt_tour)
        all_solved_costs = [tsp_solver.evaluate(solved_tours[i]) for i in range(self.parallel_sampling)]
        best_solved_cost, best_id = np.min(all_solved_costs), np.argmin(all_solved_costs)
        gap = (best_solved_cost - gt_cost) / gt_cost * 100

        # Gradient search
        if self.gradient_search:
            # select the best tour
            g_best_tour = solved_tours[best_id]  # [N+1] ndarray
            g_best_solved_cost = best_solved_cost

            for _ in range(self.rewrite_steps):
                g_stacked_tours = []
                # optimal adjacent matrix
                g_x0 = self.tour2adj(g_best_tour, np_points, self.sparse, self.sparse_factor, original_edge_index)
                g_x0 = g_x0.unsqueeze(0).to(device)  # [1, N, N] or [1, N]
                if self.parallel_sampling > 1:
                    if not self.sparse:
                        g_x0 = g_x0.repeat(self.parallel_sampling, 1, 1)  # [1, N ,N] -> [B, N, N]
                    else:
                        g_x0 = g_x0.repeat(self.parallel_sampling, 1)

                if self.sparse:
                    g_x0 = g_x0.reshape(-1)

                g_x0_onehot = F.one_hot(g_x0.long(), num_classes=2).float()  # [B, N, N, 2]
                steps_T = int(self.diffusion_steps * self.rewrite_ratio)
                time_schedule = InferenceSchedule(inference_schedule=self.inference_schedule,
                                                    T=steps_T, inference_T=self.steps_inf)
                
                # g_xt = self.diffusion.sample(g_x0_onehot, steps_T)
                Q_bar = torch.from_numpy(self.diffusion.Q_bar[steps_T]).float().to(g_x0_onehot.device)
                g_xt_prob = torch.matmul(g_x0_onehot, Q_bar)  # [B, N, N, 2]

                # add noise for the steps_T samples, namely rewrite
                g_xt = torch.bernoulli(g_xt_prob[..., 1].clamp(0, 1))  # [B, N, N]
                g_xt = g_xt * 2 - 1  # project to [-1, 1]
                g_xt = g_xt * (1.0 + 0.05 * torch.rand_like(g_xt))  # add noise
                g_xt = (g_xt > 0).long()

                for i in range(self.steps_inf):
                    t1, t2 = time_schedule(i)
                    t1 = np.array([t1]).astype(int)
                    t2 = np.array([t2]).astype(int)

                    # [1, N, N], denoise, heatmap for edges
                    g_xt = self.guided_categorical_denoise_step(points, g_xt, t1, device, edge_index, target_t=t2)

                g_adj_mat = g_xt.float().cpu().detach().numpy() + 1e-6

                # decode
                decoding_func = get_decoding_func(task="tsp", name=self.test_decoding_type)
                g_solved_tours = decoding_func(
                    adj_mat=g_adj_mat, 
                    np_points=np_points, 
                    edge_index_np=np_edge_index, 
                    sparse_graph=self.sparse, 
                    device=gt_tour.device,
                    parallel_sampling=self.parallel_sampling,
                    **self.test_decoding_kwargs
                )

                # local_search
                local_search_func = get_local_search_func(task="tsp", name=self.test_ls_type)
                if local_search_func is not None:
                    if self.sparse:
                        if self.parallel_sampling == 1:
                            sparse_g_adj_mat = SparseTensor(
                                row=edge_index[0],
                                col=edge_index[1],
                                value=torch.tensor(g_adj_mat).to(device=edge_index.device)
                            )
                            g_adj_mat = sparse_g_adj_mat.to_dense().unsqueeze(dim=0).cpu().numpy()
                        else:
                            sparse_g_adj_mat = list()
                            ps_g_edge_index = edge_index.reshape(2, self.parallel_sampling, -1).transpose(0, 1)
                            ps_g_adj_mat = g_adj_mat.reshape(self.parallel_sampling, -1)
                            for idx in range(self.parallel_sampling):
                                ps_sparse_g_adj_mat = SparseTensor(
                                    row=ps_g_edge_index[idx][0] - self.num_nodes*idx,
                                    col=ps_g_edge_index[idx][1] - self.num_nodes*idx,
                                    value=torch.tensor(ps_g_adj_mat[idx]).to(device=edge_index.device)
                                )
                                sparse_g_adj_mat.append(ps_sparse_g_adj_mat.to_dense().unsqueeze(dim=0).cpu().numpy()[0])
                            g_adj_mat = np.array(sparse_g_adj_mat)

                    g_solved_tours = local_search_func(
                        np_points=np_points, 
                        tours=g_solved_tours, 
                        adj_mat=g_adj_mat, 
                        device=gt_tour.device,
                        **self.test_ls_kwargs
                    )
                g_stacked_tours.append(g_solved_tours)

                g_solved_tours = np.concatenate(g_stacked_tours, axis=0)

                g_total_sampling = self.parallel_sampling
                g_all_solved_costs = [tsp_solver.evaluate(g_solved_tours[i]) for i in range(g_total_sampling)]
                g_best_solved_cost_tmp, g_best_id = np.min(g_all_solved_costs), np.argmin(g_all_solved_costs)
                if g_best_solved_cost_tmp < g_best_solved_cost:
                    g_best_tour = g_solved_tours[g_best_id]
                g_best_solved_cost = min(g_best_solved_cost, g_best_solved_cost_tmp)

                guided_gap = (g_best_solved_cost - gt_cost) / gt_cost * 100
            
            # print(gap, guided_gap)

            gap = guided_gap
            best_solved_cost = g_best_solved_cost


        # record the better/worse/match
        better = 0.0
        match = 0.0
        worse = 0.0
        if gap < -1e-12:
            better = 1.0
        elif gap < 1e-12:
            match = 1.0
        else:
            worse = 1.0
        self.gap_list.append(gap)

        # Log the loss and gap
        metrics = {
            f"{phase}/loss": loss,
            f"{phase}/gap": gap,
            f"{phase}/better": better,
            f"{phase}/match": match,
            f"{phase}/worse": worse,
        }

        if phase == 'test':
            metrics.update({
                "test/gt_cost": gt_cost,
                "test/solved_cost": best_solved_cost
            })

        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
        return metrics

    def solve(
        self, 
        data: Union[np.ndarray, torch.Tensor], 
        edge_index: Union[np.ndarray, torch.Tensor]=None,
        batch_size: int=16, 
        device='cpu'
    ):
        """solve function, return heatmap"""
        self.model.eval()
        if type(data) == np.ndarray:
            data = torch.Tensor(data)
        if self.sparse:
            if type(edge_index) == np.ndarray:
                edge_index = torch.Tensor(edge_index)
            edge_index = edge_index.to(device)
    
        assert data.shape[0] % batch_size == 0, f'batch_size must be divided by the size of the data'
        
        if not self.sparse:
            batch_data = data.reshape(-1, batch_size, data.shape[-2], 2)
            batch_edge_index = [None] * batch_data.shape[0]
        else:
            batch_num = data.shape[0] // batch_size
            batch_data = data.reshape(batch_num, -1, 2)
            batch_edge_index = edge_index.reshape(batch_num, -1, 2, edge_index.shape[-1])
            batch_edge_index = batch_edge_index.transpose(1, 2)
            batch_edge_index = batch_edge_index.reshape(batch_num, 2, -1)
            length = self.sparse_factor * self.num_nodes
            for idx in range(batch_size):
                batch_edge_index[:, :, idx*length:(idx+1)*length] += (idx * self.num_nodes) 
        batch_heatmap = list()
        
        with torch.no_grad():
            for idx, points in tqdm(enumerate(batch_data), total=batch_data.shape[0], desc='Processing'):
                if self.parallel_sampling > 1:
                    if not self.sparse:
                        points = points.repeat(self.parallel_sampling, 1, 1)
                    else:
                        points = points.repeat(self.parallel_sampling, 1)
                        edge_index = self.duplicate_edge_index(batch_edge_index[idx], self.num_nodes, device)
                else:
                    edge_index = batch_edge_index[idx]
                
                if self.sparse:
                    xt = torch.randn(self.num_nodes * self.sparse_factor)
                else:
                    xt = torch.randn(self.num_nodes, self.num_nodes)
                xt = xt.to(points.device)
                
                if self.parallel_sampling > 1:
                    if not self.sparse:
                        xt = xt.repeat(self.parallel_sampling, 1, 1)
                    else:
                        xt = xt.repeat(self.parallel_sampling, 1)  # [B, E]
                    xt = torch.randn_like(xt)
                xt = (xt > 0).long()

                if self.sparse:
                    xt = xt.reshape(-1)

                steps = self.inference_diffusion_steps
                time_schedule = InferenceSchedule(inference_schedule=self.inference_schedule,
                                                T=self.diffusion.T, inference_T=steps)
                
                # Diffusion iterations
                for i in range(steps):
                    t1, t2 = time_schedule(i)
                    t1 = np.array([t1]).astype(int)
                    t2 = np.array([t2]).astype(int)
                    # [B, N, N], heatmap score
                    xt = self.categorical_denoise_step(
                        points, xt, t1, device, edge_index, target_t=t2)

                batch_heatmap.append(xt.cpu().numpy())

        batch_heatmap = np.array(batch_heatmap)
        if not self.sparse:
            heatmap = batch_heatmap.reshape(-1, batch_heatmap.shape[-2], batch_heatmap.shape[-1])
        else:
            heatmap = batch_heatmap.reshape(-1, batch_heatmap.shape[-1])
        return heatmap

    def load_ckpt(self, ckpt_path: str=None):
        """load state dict from checkpoint"""
        if ckpt_path is None:
            if self.num_nodes in [50, 100, 500]:
                url = tsp_diffusion_path[self.num_nodes]
                filename=f"ckpts/tsp{self.num_nodes}_diffusion.pt"
                pygm.utils.download(filename=filename, url=url, to_cache=None)
                self.load_state_dict(torch.load(filename))
            else:
                raise ValueError(f"There is currently no pretrained checkpoint with {self.num_nodes} nodes.")
        else:
            checkpoint = torch.load(ckpt_path)
            if ckpt_path.endswith('.ckpt'):
                state_dict = checkpoint['state_dict']
            elif ckpt_path.endswith('.pt'):
                state_dict = checkpoint
            self.load_state_dict(state_dict)

    def categorical_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)

            ###############################################
            # scale to [-1, 1]
            xt_scale = (xt * 2 - 1).float()
            xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt_scale))
            # xt_scale = xt
            ###############################################
            x0_pred = self.forward(
                points.float().to(device),
                xt_scale.float().to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )
            if not self.sparse:
                if self.network_type == 'gnn':
                    x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
                else: # for gat encoder
                    x0_pred_prob = x0_pred.view(-1, self.num_nodes, self.num_nodes, self.out_channels).softmax(dim=-1)
            else:
                x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)

            xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
            return xt
    
    @torch.enable_grad() 
    @torch.inference_mode(False)
    def guided_categorical_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
        xt = xt.float()  # b, n, n
        xt.requires_grad = True
        t = torch.from_numpy(t).view(1)
        if edge_index is not None: edge_index = edge_index.clone()

        # [b, 2, n, n]
        # with torch.inference_mode(False):
        ###############################################
        # scale to [-1, 1]
        xt_scale = (xt * 2 - 1)
        xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt_scale))
        # xt_scale = xt
        ###############################################

        # print(points.shape, xt.shape)
        x0_pred = self.forward(
            points.float().to(device),
            xt_scale.to(device),
            t.float().to(device),
            edge_index.long().to(device) if edge_index is not None else None,
        )

        if not self.sparse:
            x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
        else:
            x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)

        if not self.sparse:
            dis_matrix = self.points2adj(points)
            cost_est = (dis_matrix * x0_pred_prob[..., 1]).sum()
            cost_est.requires_grad_(True)
            cost_est.backward()
        else:
            dis_matrix = torch.sqrt(torch.sum((points[edge_index.T[:, 0]] - points[edge_index.T[:, 1]]) ** 2, dim=1))
            dis_matrix = dis_matrix.reshape((1, points.shape[0], -1))
            cost_est = (dis_matrix * x0_pred_prob[..., 1]).sum()
            cost_est.requires_grad_(True)
            cost_est.backward()
        assert xt.grad is not None

        xt.grad = nn.functional.normalize(xt.grad, p=2, dim=-1)
        xt = self.guided_categorical_posterior(target_t, t, x0_pred_prob, xt)

        return xt.detach()
        
    def duplicate_edge_index(self, edge_index, num_nodes, device):
        """Duplicate the edge index (in sparse graphs) for parallel sampling."""
        edge_index = edge_index.reshape((2, 1, -1))
        edge_index_indent = torch.arange(0, self.parallel_sampling).view(1, -1, 1).to(device)
        edge_index_indent = edge_index_indent * num_nodes
        edge_index = edge_index + edge_index_indent
        edge_index = edge_index.reshape((2, -1))
        return edge_index

    def tour2adj(self, tour, points, sparse, sparse_factor, edge_index):
        if not sparse:
            adj_matrix = torch.zeros((points.shape[0], points.shape[0]))
            for i in range(tour.shape[0] - 1):
                adj_matrix[tour[i], tour[i + 1]] = 1
        else:
            adj_matrix = np.zeros(points.shape[0], dtype=np.int64)
            adj_matrix[tour[:-1]] = tour[1:]
            adj_matrix = torch.from_numpy(adj_matrix)
            adj_matrix = adj_matrix.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
            adj_matrix = torch.eq(edge_index[1], adj_matrix).to(torch.int)
        return adj_matrix

    def points2adj(self, points):
        """
        return distance matrix
        Args:
        points: b, n, 2
        Returns: b, n, n
        """
        assert points.dim() == 3
        return torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1) ** 0.5