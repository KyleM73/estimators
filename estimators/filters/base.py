from typing import Type

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # mps is almost always slower
torch.manual_seed(0)

from estimators.dynamics import BaseDynamics

class BaseFilter:
    def __init__(self, dynamics : Type[BaseDynamics] | None = None, logging : bool = True, device : str = DEVICE, **kwargs) -> None :
        assert dynamics is not None, f"[{self.__class__.__name__}] Dynamics not specified"
        self.dynamics = dynamics
        self.logging = logging
        self.device = device

    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"[{self.__class__.__name__}] __call__() not implemented")