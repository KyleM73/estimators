import torch
import numpy as np

from estimators.dynamics import BaseDynamics

class ParticleDynamics(BaseDynamics):
    def __init__(self, dim : int = 1, **kwargs) -> None:
        super(ParticleDynamics, self).__init__(**kwargs)
        self.dim = dim
        self.x0 = torch.zeros(2 * self.dim, 1, dtype=torch.float64, device=self.device)
        self.A = torch.from_numpy(
            np.block([[np.zeros((self.dim, self.dim)), np.eye(self.dim)], [np.zeros((self.dim, 2*self.dim))]])
            ).to(dtype=torch.float64, device=self.device)
        self.A_d = torch.eye(2 * self.dim, 2 * self.dim, device=self.device) + self.dt * self.A
        self.B = torch.from_numpy(
            np.block([[np.zeros((self.dim, self.dim))], [np.eye(self.dim)]])
            ).to(dtype=torch.float64, device=self.device)
        self.B_d = self.dt * self.B
        self.reset()

    def reset(self) -> None:
        self.x = self.x0
        
    def step(self, u : torch.Tensor | None = None) -> torch.Tensor:
        if u is None: u = torch.zeros(self.dim, 1, dtype=torch.float64, device=self.device)
        else: u.view(self.dim, 1).to(dtype=torch.float64, device=self.device)
        self.x = self.rk4(self.A, self.B, self.x, u)
        return self.x
    
    def step_discrete(self, x : torch.Tensor, u : torch.Tensor | None = None) -> torch.Tensor:
        if u is None: u = torch.zeros(self.dim, 1, dtype=torch.float64, device=self.device)
        else: u.view(self.dim, 1).to(dtype=torch.float64, device=self.device)
        return torch.matmul(self.A_d, x) + torch.matmul(self.B_d, u)
    
    def get_x0(self) -> torch.Tensor:
        return self.x0
    
    def set_x0(self, x0 : torch.Tensor) -> None:
        self.x0 = x0.to(dtype=torch.float64, device=self.device)


