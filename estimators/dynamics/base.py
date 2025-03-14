import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # mps is almost always slower
torch.manual_seed(0)

class BaseDynamics:
    def __init__(self, dt : float = 0.1, device : str = DEVICE, **kwargs) -> None:
        self.dt = dt
        self.device = device
        print(f"[{self.__class__.__name__}] Initialized")

    def reset(self) -> None:
        raise NotImplementedError(f"[{self.__class__.__name__}] reset() not implemented")
    
    def step(self, u : torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"[{self.__class__.__name__}] step() not implemented")
    
    def rk4(self, A : torch.Tensor, B : torch.Tensor, x : torch.Tensor, u : torch.Tensor) -> torch.Tensor:
        k1 = torch.matmul(A, x) + torch.matmul(B, u)
        k2 = torch.matmul(A, (x + self.dt * k1 / 2)) + torch.matmul(B, u)
        k3 = torch.matmul(A,(x + self.dt * k2 / 2)) + torch.matmul(B, u)
        k4 = torch.matmul(A,(x + self.dt * k3)) + torch.matmul(B, u)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6

