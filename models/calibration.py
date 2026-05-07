"""
Temperature Scaling Calibration — SynFeasNet v2
==================================================
Post-training calibration using a single learned temperature parameter.

WHY:
  Neural networks are often overconfident. Temperature scaling
  divides logits by T before sigmoid, spreading the probability
  distribution. T is fit on the validation set using NLL.

  P_calibrated = sigmoid(logit / T)

Usage:
    calibrator = TemperatureScaling()
    calibrator.calibrate(val_logits, val_labels)
    calibrated_prob = calibrator.calibrated_probability(test_logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for probability calibration.
    Learns a single scalar T on the validation set.
    """

    def __init__(self, init_temp: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temp]))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature

    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor,
                  lr: float = 0.01, max_iter: int = 100) -> float:
        """
        Fit temperature on validation logits/labels.

        Args:
            logits: (N,) or (N,1) raw model logits.
            labels: (N,) binary labels.

        Returns:
            Fitted temperature value.
        """
        logits = logits.detach().float().view(-1)
        labels = labels.detach().float().view(-1)

        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter
        )

        def closure():
            optimizer.zero_grad()
            scaled = logits / self.temperature
            loss = F.binary_cross_entropy_with_logits(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"  Calibration temperature: {self.temperature.item():.4f}")
        return self.temperature.item()

    def calibrated_probability(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling and return calibrated probabilities."""
        return torch.sigmoid(self.forward(logits))


def compute_ece(probs: np.ndarray, labels: np.ndarray,
                n_bins: int = 15) -> float:
    """
    Expected Calibration Error.
    Lower = better calibrated.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(probs))
