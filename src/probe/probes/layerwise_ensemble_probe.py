from collections.abc import Callable

import torch
from torch import nn


class LayerwiseEnsembleProbe(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        probe_factory: Callable[[], nn.Module],
        aggregation: str = "weighted_logits",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("LayerwiseEnsembleProbe num_layers must be >= 1.")
        if aggregation not in {"weighted_logits", "per_layer"}:
            raise ValueError(
                "LayerwiseEnsembleProbe aggregation must be 'weighted_logits' "
                f"or 'per_layer', got {aggregation!r}."
            )
        self.aggregation = aggregation
        self.probes = nn.ModuleList([probe_factory() for _ in range(num_layers)])
        self.logit_aggregator = (
            nn.Linear(num_layers, 1) if aggregation == "weighted_logits" else None
        )
        if self.logit_aggregator is not None:
            nn.init.constant_(self.logit_aggregator.weight, 1.0 / float(num_layers))
            nn.init.zeros_(self.logit_aggregator.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "LayerwiseEnsembleProbe expects [batch, layer, hidden] inputs, "
                f"got shape {tuple(x.shape)}"
            )
        if x.size(1) != len(self.probes):
            raise ValueError(
                "Layer count mismatch between features and ensemble probes: "
                f"{x.size(1)} vs {len(self.probes)}"
            )
        logits = [probe(x[:, idx, :]) for idx, probe in enumerate(self.probes)]
        layer_logits = torch.stack(logits, dim=1)
        if self.logit_aggregator is None:
            return layer_logits
        return self.logit_aggregator(layer_logits).squeeze(-1)
