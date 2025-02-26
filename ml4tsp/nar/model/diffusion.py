import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Union, Tuple
from ..env import ML4TSPNAREnv
from ..decoder import ML4TSPNARDecoder
from ..model.base import ML4TSPNARBaseModel
from ..local_search import ML4TSPNARLocalSearch
from ..encoder.gnn.gnn_encoder import GNNEncoder
import math


class ML4TSPDiffusion(ML4TSPNARBaseModel):
    def __init__(
        self,
        env: ML4TSPNAREnv,
        encoder: GNNEncoder,
        decoder: Union[ML4TSPNARDecoder, str] = "greedy",
        local_search: Union[ML4TSPNARLocalSearch, str] = None,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        pretrained_path: str = None,
        diffusion_schedule: str="linear",
        inference_schedule: str="cosine",
        diffusion_steps: int=1000,
        inference_diffusion_steps: int=1,
    ):
        # super
        super(ML4TSPDiffusion, self).__init__(
            env=env,
            encoder=encoder,
            decoder=decoder,
            local_search=local_search,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pretrained=pretrained,
            pretrained_path=pretrained_path
        )
        self.diffusion_schedule = diffusion_schedule
        self.inference_schedule = inference_schedule
        self.diffusion_steps = diffusion_steps
        self.inference_diffusion_steps = inference_diffusion_steps
        self.diffusion = CategoricalDiffusion(
            T=self.diffusion_steps, schedule=self.diffusion_schedule)
    
    def inference_process(
        self, points: Tensor, edge_index: Tensor, distmat: Tensor, ground_truth: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        
        batch_size = points.shape[0]
        if self.env.sparse:
            points = points.reshape(-1, 2)
            distmat = distmat.reshape(-1)
            edge_index = edge_index.transpose(1, 0).reshape(2, -1)
            x_shape = (batch_size, edge_index.shape[1] // batch_size)
        else:
            x_shape = (batch_size, points.shape[1], points.shape[1])
        
        xt = torch.randn(x_shape).to(points.device)
        
        xt = (xt > 0).long()
        if self.env.sparse:
            xt = xt.reshape(-1)
        
        steps = self.inference_diffusion_steps
        time_schedule = InferenceSchedule(
            inference_schedule=self.inference_schedule,
            T=self.diffusion.T, inference_T=steps
        )
        
        for i in range(steps):
            t1, t2 = time_schedule(i)
            t1 = np.array([t1]).astype(int)
            t2 = np.array([t2]).astype(int)
        
            # [B, N, N], heatmap score
            xt, x0_pred = self.categorical_denoise_step(
                points, xt, t1, points.device, edge_index, target_t=t2)
        
        heatmap = xt.float().cpu().detach().numpy() + 1e-6
        
        if self.env.sparse:
            x0_pred = x0_pred.T.reshape(batch_size, 2, -1)
            
        if self.env.mode == "solve":
            return heatmap.reshape(batch_size, -1)
    
        # Compute loss
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(x0_pred, ground_truth.long())
        return loss, heatmap
    
    def train_process(
        self, points: Tensor, edge_index: Tensor, distmat: Tensor, ground_truth: Tensor
    ) -> Tensor:
        
        batch_size = points.shape[0]
        if self.env.sparse:
            points = points.reshape(-1, 2)
            distmat = distmat.reshape(-1)
            edge_index = edge_index.transpose(1, 0).reshape(2, -1)
        
        # Sample from diffusion
        adj_matrix_onehot = F.one_hot(ground_truth.long(), num_classes=2).float()
        if self.env.sparse:
            adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1)
            
        np_t = np.random.randint(1, self.diffusion.T + 1, batch_size).astype(int)
        if self.env.sparse:
            t = torch.from_numpy(np_t).float().reshape(-1, 1).repeat(1, ground_truth.shape[1]).reshape(-1)
        else:
            t = torch.from_numpy(np_t).float().view(ground_truth.shape[0])
            
        xt = self.diffusion.sample(adj_matrix_onehot, np_t)
        xt = xt * 2 - 1
        xt = xt * (1.0 + 0.05 * torch.rand_like(xt))
        
        if self.env.sparse:
            xt = xt.reshape(-1)

        # x0_pred
        x0_pred = self.forward(
            x=points, 
            graph=xt.float().to(ground_truth.device), 
            edge_index=edge_index, 
            timesteps=t.float().to(ground_truth.device)
        )
        
        if self.env.sparse:
            x0_pred = x0_pred.T.reshape(batch_size, 2, -1)
        
        # Compute loss
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(x0_pred, ground_truth.long())
        
        return loss
    
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
                x=points.float().to(device),
                graph=xt_scale.float().to(device),
                timesteps=t.float().to(device),
                edge_index=edge_index.long().to(device) if edge_index is not None else None,
            )
            if not self.env.sparse:
                x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
            else:
                x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)

            xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
            return xt, x0_pred
    
    def categorical_posterior(self, target_t, t, x0_pred_prob, xt):
        """Sample from the categorical posterior for a given time step.
        See https://arxiv.org/pdf/2107.03006.pdf for details.
        """
        diffusion = self.diffusion
        
        if target_t is None:
            target_t = t - 1
        else:
            target_t = torch.from_numpy(target_t).view(1)

        if target_t > 0:
            Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
            Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)
        else:
            Q_t = torch.eye(2).float().to(x0_pred_prob.device)
        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)

        xt = F.one_hot(xt.long(), num_classes=2).float()
        xt = xt.reshape(x0_pred_prob.shape)
        
        x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
        x_t_target_prob_part_2 = Q_bar_t_target[0]
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]
        x_t_target_prob_part_2_new = Q_bar_t_target[1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

        x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new

        sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

        if target_t > 0:
            xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
        else:
            xt = sum_x_t_target_prob.clamp(min=0)

        if self.env.sparse:
            xt = xt.reshape(-1)
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
        
        x0_pred = self.forward(
            x=points.float().to(device),
            graph=xt_scale.to(device),
            timesteps=t.float().to(device),
            edge_index=edge_index.long().to(device) if edge_index is not None else None,
        )

        if not self.env.sparse:
            x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
        else:
            x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)

        if not self.env.sparse:
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
    
    def guided_categorical_posterior(self, target_t, t, x0_pred_prob, xt, grad=None):
        # xt: b, n, n
        if grad is None:
            grad = xt.grad
        with torch.no_grad():
            diffusion = self.diffusion
        if target_t is None:
            target_t = t - 1
        else:
            target_t = torch.from_numpy(target_t).view(1)

        if target_t > 0:
            Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
            Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)  # [2, 2], transition matrix
        else:
            Q_t = torch.eye(2).float().to(x0_pred_prob.device)
        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)

        xt_grad_zero, xt_grad_one = torch.zeros(xt.shape, device=xt.device).unsqueeze(-1).repeat(1, 1, 1, 2), \
            torch.zeros(xt.shape, device=xt.device).unsqueeze(-1).repeat(1, 1, 1, 2)
        xt_grad_zero[..., 0] = (1 - xt) * grad
        xt_grad_zero[..., 1] = -xt_grad_zero[..., 0]
        xt_grad_one[..., 1] = xt * grad
        xt_grad_one[..., 0] = -xt_grad_one[..., 1]
        xt_grad = xt_grad_zero + xt_grad_one

        xt = F.one_hot(xt.long(), num_classes=2).float()
        xt = xt.reshape(x0_pred_prob.shape)  # [b, n, n, 2]

        # q(xt−1|xt,x0=0)pθ(x0=0|xt)
        x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
        x_t_target_prob_part_2 = Q_bar_t_target[0]
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3  # [b, n, n, 2]

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]

        # q(xt−1|xt,x0=1)pθ(x0=1|xt)
        x_t_target_prob_part_2_new = Q_bar_t_target[1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

        x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new

        sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

        p_theta = torch.cat((1 - sum_x_t_target_prob.unsqueeze(-1), sum_x_t_target_prob.unsqueeze(-1)), dim=-1)
        p_phi = torch.exp(-xt_grad)
        if self.env.sparse:
            p_phi = p_phi.reshape(p_theta.shape)
        posterior = (p_theta * p_phi) / torch.sum((p_theta * p_phi), dim=-1, keepdim=True)

        if target_t > 0:
            xt = torch.bernoulli(posterior[..., 1].clamp(0, 1))
        else:
            xt = posterior[..., 1].clamp(min=0)
        if self.env.sparse:
            xt = xt.reshape(-1)
        return xt
    
    def points2adj(self, points):
        """
        return distance matrix
        Args:
        points: b, n, 2
        Returns: b, n, n
        """
        assert points.dim() == 3
        return torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1) ** 0.5
    
    
class CategoricalDiffusion(object):
  def __init__(self, T, schedule):
    # Diffusion steps
    self.T = T

    # Noise schedule
    if schedule == 'linear':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0, bT, T)
    elif schedule == 'cosine':
      self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
          0)  # Generate an extra alpha for bT
      self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

    beta = self.beta.reshape((-1, 1, 1))
    eye = np.eye(2).reshape((1, 2, 2))
    ones = np.ones((2, 2)).reshape((1, 2, 2))

    self.Qs = (1 - beta) * eye + (beta / 2) * ones

    Q_bar = [np.eye(2)]
    for Q in self.Qs:
      Q_bar.append(Q_bar[-1] @ Q)
    self.Q_bar = np.stack(Q_bar, axis=0)

  def __cos_noise(self, t):
    offset = 0.008
    return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

  def sample(self, x0_onehot, t):
    # Select noise scales
    Q_bar = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)
    xt = torch.matmul(x0_onehot, Q_bar.reshape((Q_bar.shape[0], 1, 2, 2)))
    return torch.bernoulli(xt[..., 1].clamp(0, 1))


class InferenceSchedule(object):
  def __init__(self, inference_schedule="linear", T=1000, inference_T=1000):
    self.inference_schedule = inference_schedule
    self.T = T
    self.inference_T = inference_T

  def __call__(self, i):
    assert 0 <= i < self.inference_T

    if self.inference_schedule == "linear":
      t1 = self.T - int((float(i) / self.inference_T) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      return t1, t2
    elif self.inference_schedule == "cosine":
      t1 = self.T - int(
          np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int(
          np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      return t1, t2
    else:
      raise ValueError("Unknown inference schedule: {}".format(self.inference_schedule))
