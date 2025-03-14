from dataclasses import dataclass, MISSING

import torch

@dataclass
class KalmanFilterCfg:
    A: torch.Tensor = MISSING
    """State Transition Matrix"""

    B: torch.Tensor = MISSING
    """Control Input Matrix"""

    C: torch.Tensor = MISSING
    """Measurement/Observation Matrix"""

    Q: torch.Tensor = MISSING
    """Process Noise Covariance Matrix"""

    R: torch.Tensor = MISSING
    """Measurement Noise Covariance Matrix"""


class KalmanFilter:
    """
    Kalman Filter
    
    Assumptions:
    -   LTI system dynamics
    -   Discrete time
    -   Gaussian white noise
    -   Gaussian prior state distribution

    see: https://en.wikipedia.org/wiki/Kalman_filter
    """
    def __init__(
            self,
            cfg: KalmanFilterCfg,
            x_prior: torch.Tensor,
            P_prior: torch.Tensor,
            device: str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = device

        # Validate config
        self._validate_covariance(cfg.Q, cfg.R)
        self._validate_LTI_system(cfg.A, cfg.B, cfg.C)

        # Ensure tensors on correct device
        self.x = x_prior.to(self.device)
        self.P = P_prior.to(self.device)
        for key, value in vars(self.cfg).items():
            setattr(self.cfg, key, value.to(self.device))


    @property
    def num_states(self) -> int:
        return self.cfg.A.size(0)

    @property
    def num_controls(self) -> int:
        return self.cfg.B.size(1)
    
    @property
    def num_measurements(self) -> int:
        return self.cfg.C.size(0)
    
    @property
    def I(self) -> torch.Tensor:
        return torch.eye(self.num_states, device=self.device)

    def _predict(
            self,
            u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the estimate and covariance via the state transition dynamics."""
        # A priori state estimate
        x_pred = self.cfg.A @ self.x + self.cfg.B @ u
        # A priori estimate covariance
        P_pred = self.cfg.A @ self.P @ self.cfg.A.T + self.cfg.Q
        return x_pred, P_pred
    
    def _update(
            self,
            z: torch.Tensor,
            x_pred: torch.Tensor,
            P_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate the observation/measurement to update the estimate and covariance."""
        # Measurement pre-fit residual
        y = z - self.cfg.C @ x_pred
        # Pre-fit residual covariance
        S = self.cfg.C @ P_pred @ self.cfg.C.T + self.cfg.R
        # Kalman gain
        K = P_pred @ self.cfg.C.T @ torch.linalg.pinv(S)
        # A posteriori state estimate
        x_est = x_pred + K @ y
        # A posteriori estimate covariance
        P_est = (self.I - K @ self.cfg.C) @ P_pred
        return x_est, P_est
    
    def step(
            self,
            u: torch.Tensor,
            z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u, z = u.to(self.device), z.to(self.device)
        x_pred, P_pred = self._predict(u)
        self.x, self.P = self._update(z, x_pred, P_pred)
        return self.x.clone(), self.P.clone()
    
    @staticmethod
    def _validate_covariance(Q: torch.Tensor, R: torch.Tensor) -> None:
        assert torch.allclose(Q, Q.T, atol=1e-6), "Q must be symmetric."
        assert torch.all(torch.linalg.eigvalsh(Q) >= 0), "Q must be positive semi-definite."
        assert torch.allclose(R, R.T, atol=1e-6), "R must be symmetric."
        assert torch.all(torch.linalg.eigvalsh(R) >= 0), "R must be positive semi-definite."
        
    @staticmethod
    def _validate_LTI_system(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
        assert A.shape[0] == A.shape[1], "A must be square."
        assert B.shape[0] == A.shape[0], "B must have same rows as A."
        assert C.shape[1] == A.shape[1], "C must have same columns as A."