# Standard libraries
from typing import List

# Third-party libraries
import torch
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupScheduler(_LRScheduler):
    def __init__(
        self, optimizer: torch.optim.Optimizer, total_steps: int = 500
    ) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must inherit from torch.optim.Optimizer")
        self.finished = False
        self.total_steps = total_steps
        super(LinearWarmupScheduler, self).__init__(optimizer)

    def attach_scheduler(
        self, scheduler: torch.optim.lr_scheduler._LRScheduler
    ) -> None:
        if not isinstance(scheduler, _LRScheduler):
            raise TypeError("scheduler must inherit from _LRScheduler")
        self.attached_scheduler = scheduler

    def get_lr(self) -> List[float]:
        if self._step_count > self.total_steps:
            if self.attached_scheduler:
                if not self.finished:
                    self.attached_scheduler.base_lrs = [
                        base_lr for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.attached_scheduler.get_last_lr()
            return [base_lr for base_lr in self.base_lrs]
        else:
            return [
                base_lr * (float(self._step_count) / self.total_steps)
                for base_lr in self.base_lrs
            ]

    def step(self) -> None:
        if self.finished and self.attached_scheduler:
            self.attached_scheduler.step()
        else:
            super(LinearWarmupScheduler, self).step()
