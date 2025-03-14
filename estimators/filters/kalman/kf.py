from typing import List

import torch
from estimators.dynamics import ParticleDynamics
from estimators.filters import BaseFilter

class KalmanFilter(BaseFilter):
    def __init__(self, Q : torch.Tensor, R : torch.Tensor, H : torch.Tensor | None = None, **kwargs) -> None:
        super(KalmanFilter, self).__init__(**kwargs)
        self.Q = Q.to(dtype=torch.float64, device=self.device)
        self.N_Q = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(2 * self.dynamics.dim, dtype=torch.float64, device=self.device),
            covariance_matrix=self.Q
            )
        self.R = R.to(dtype=torch.float64, device=self.device)
        self.N_R = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(2 * self.dynamics.dim, dtype=torch.float64, device=self.device),
            covariance_matrix=self.R
            )
        self.H = torch.eye(2 * self.dynamics.dim, dtype=torch.float64, device=self.device) if H is None else H
        self.P = torch.eye(2 * self.dynamics.dim, dtype=torch.float64, device=self.device)
        self.x = self.dynamics.get_x0()
        self.steps = 0
        self.logs = {
            "pre_fit_x" : [],
            "pre_fit_P" : [],
            "post_fit_x" : [],
            "post_fit_P" : [],
            "x" : [],
            }
        
    def __call__(self, u : torch.Tensor | None = None) -> torch.Tensor:
        # predict
        x_true = self.dynamics.step(u)
        x_pred = self.dynamics.step_discrete(self.x, u)
        P_pred = self.dynamics.A_d * self.P * torch.transpose(self.dynamics.A_d, 0, 1) + self.Q
        
        #update
        z = torch.matmul(self.H, x_true) + self.N_R.sample().view(2 * self.dynamics.dim, 1)
        x_pre_fit_residual = z - torch.matmul(self.H, x_pred)
        P_pre_fit = torch.matmul(self.H, P_pred) * torch.transpose(self.H, 0, 1) + self.R
        K = torch.matmul(torch.matmul(P_pred, torch.transpose(self.H, 0, 1)), torch.linalg.inv(P_pre_fit))
        self.x = x_pred + torch.matmul(K, x_pre_fit_residual)
        self.P = torch.matmul((torch.eye(2 * self.dynamics.dim, dtype=torch.float64, device=self.device) - torch.matmul(K, self.H)), P_pred)
        x_post_fit_residual = z - torch.matmul(self.H, self.x)

        if self.logging:
            self.logs["pre_fit_x"].append(x_pre_fit_residual[..., None])
            self.logs["pre_fit_P"].append(P_pre_fit[..., None])
            self.logs["post_fit_x"].append(x_post_fit_residual[..., None])
            self.logs["post_fit_P"].append(self.P[..., None])
            self.logs["x"].append(self.x[..., None])

        self.steps += 1
        return self.x
    
    def plot(self) -> None:
        assert self.logging, "[{}] Logging not enabled".format(self.__class__.__name__)
        import matplotlib.pyplot as plt
        fig_x,ax_x = plt.subplots(2 * self.dynamics.dim)
        fig_resid,ax_resid = plt.subplots(2 * self.dynamics.dim)
        fig_P,ax_P = plt.subplots()
        for k,v in self.logs.items():
            data = torch.cat(v, dim=2).cpu()
            if "_x" in k:
                fig_resid.suptitle("Residuals", fontsize=16)
                for i in range(2 * self.dynamics.dim):
                    ax_resid[i].plot(data[i, 0, :], label="x{i} {n}".format(i=i,n=k[:-2]))
                    ax_resid[i].legend(loc="upper right")
                    if i < 2 * self.dynamics.dim - 1: ax_resid[i].get_xaxis().set_visible(False)
            elif "x" in k:
                fig_x.suptitle("State", fontsize=16)
                for i in range(2 * self.dynamics.dim):
                    ax_x[i].plot(data[i, 0, :], label="x{}".format(i))
                    ax_x[i].legend(loc="upper right")
                    if i < 2 * self.dynamics.dim - 1: ax_x[i].get_xaxis().set_visible(False)
            elif "_P" in k:
                fig_P.suptitle("Covariances", fontsize=16)
                ax_P.plot(torch.linalg.norm(data, dim=(0,1)), label="norm(P) {}".format(k[:-2]))
                ax_P.legend(loc="upper right")
        plt.show()

if __name__ == "__main__":
    import math
    N = 1000
    dim = 1
    dyn = ParticleDynamics(dim=dim)
    Q = 0.1 * torch.eye(2*dim)
    R = 0.01 * torch.eye(2*dim)
    kf = KalmanFilter(Q=Q, R=R, dynamics=dyn)
    for i in range(N):
        u = torch.cos(torch.tensor(i*2*math.pi/N)).view(dim, 1).to(torch.float64)
        x = kf(u)
    kf.plot()
