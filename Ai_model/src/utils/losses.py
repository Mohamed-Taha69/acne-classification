import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)  # p_t
        focal_weight = (1 - pt).pow(self.gamma)

        ce = F.nll_loss(log_probs, targets, weight=self.weight, reduction='none')
        loss = focal_weight * ce

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class ClassAwareFocalLoss(nn.Module):
    def __init__(self, gamma_per_class, weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma_per_class = torch.tensor(gamma_per_class, dtype=torch.float)
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.gamma_per_class.device != logits.device:
            self.gamma_per_class = self.gamma_per_class.to(logits.device)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        # gather p_t and gamma per example
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        gamma = self.gamma_per_class[targets]
        focal_weight = (1 - pt).pow(gamma)

        ce = F.nll_loss(log_probs, targets, weight=self.weight, reduction='none')
        loss = focal_weight * ce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class BoundaryAwareFocalLoss(nn.Module):
    """Focal loss with extra penalty for Moderate(1) <-> Severe(2) confusion"""
    def __init__(self, gamma_per_class, weight: torch.Tensor | None = None, 
                 confusion_penalty: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma_per_class = torch.tensor(gamma_per_class, dtype=torch.float)
        self.weight = weight
        self.confusion_penalty = confusion_penalty
        self.reduction = reduction
        # Moderate = 1, Severe = 2
        self.confusion_pairs = [(1, 2), (2, 1)]  # (target, confusing_class)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.gamma_per_class.device != logits.device:
            self.gamma_per_class = self.gamma_per_class.to(logits.device)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        gamma = self.gamma_per_class[targets]
        focal_weight = (1 - pt).pow(gamma)

        ce = F.nll_loss(log_probs, targets, weight=self.weight, reduction='none')
        base_loss = focal_weight * ce

        # Add confusion penalty for Moderate <-> Severe
        confusion_loss = torch.zeros_like(base_loss)
        for target_cls, confusing_cls in self.confusion_pairs:
            mask = (targets == target_cls)
            if mask.any():
                confusing_probs = probs[mask, confusing_cls]
                confusion_loss[mask] += self.confusion_penalty * confusing_probs

        total_loss = base_loss + confusion_loss
        if self.reduction == 'mean':
            return total_loss.mean()
        if self.reduction == 'sum':
            return total_loss.sum()
        return total_loss


