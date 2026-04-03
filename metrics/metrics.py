from abc import ABC, abstractmethod

from torch import nn


class Metric(ABC):
    @abstractmethod
    def update(self, predict: dict, truth: dict) -> None:
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def reset(self):
        pass
"""
class BatchedMetric(Metric):
    def __init__(self, metric_fn:nn.Module):
        self.metric_fn = metric_fn
        self.running_metric: float = 0.
        self.n_points: int = 0

    def update(self, predict:dict, truth: dict) -> None:
        batch_size = len(next(iter(predict)))
        self.n_points += batch_size

        self.running_metric += batch_size * self.metric_fn(predict, truth).item()
    
    def compute(self) -> float:
        return self.running_metric/self.n_points
    
    def reset(self) -> None:
        self.running_metric = 0
        self.n_points = 0

"""

class BatchedMetric:
    def __init__(self, metric_fn: nn.Module):
        self.metric_fn = metric_fn
        # Names of the components returned by your loss function
        self.metric_names = ["total", "position", "energy", "event"]
        
        # Initialize running totals for each component
        self.running_metrics = {name: 0.0 for name in self.metric_names}
        self.n_points = 0

    def update(self, predict: dict, truth: dict) -> None:
        # 1. Get the batch size from the first dictionary entry
        batch_size = len(next(iter(predict.values())))
        self.n_points += batch_size

        # 2. Get the tuple of 4 Tensors: (total, pos, energy, evt)
        loss_outputs = self.metric_fn(predict, truth)

        # 3. Use a list comprehension with .item() to detach from the graph
        # This prevents the AttributeError: 'tuple' has no attribute 'item'
        detached_values = [loss.item() for loss in loss_outputs]

        # 4. Update each tracker
        for name, value in zip(self.metric_names, detached_values):
            self.running_metrics[name] += batch_size * value

    def compute(self) -> dict:
        # Returns a dictionary of averages for easy logging to WandB
        if self.n_points == 0:
            return {name: 0.0 for name in self.metric_names}
            
        return {
            name: total / self.n_points 
            for name, total in self.running_metrics.items()
        }

    def reset(self) -> None:
        self.running_metrics = {name: 0.0 for name in self.metric_names}
        self.n_points = 0
