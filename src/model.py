"""
model.py — SimCLR model: ResNet backbone + projection head.

Architecture:
    Image -> ResNet18 -> feature vector (512-d) -> MLP projection head -> 128-d

Why a projection head?
    The SimCLR paper found that applying contrastive loss directly to features
    HURTS representation quality. The projection head is a "sacrificial" MLP
    that gets discarded after training. We only keep the ResNet backbone.

After training:
    - backbone.forward(x) returns representations we USE downstream
    - projection_head is THROWN AWAY
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


class ProjectionHead(nn.Module):
    """Small MLP that projects features into the contrastive space."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """ResNet backbone + projection head for SimCLR pretraining."""

    def __init__(
        self,
        backbone: str = "resnet18",
        projection_dim: int = 128,
        hidden_dim: int = 512,
    ):
        super().__init__()

        if backbone == "resnet18":
            net = tvm.resnet18(weights=None)             # ← random init, must learn everything
            # net = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)  # ← ImageNet pretrained! Done to improve the 1-NN accuracy
            feat_dim = net.fc.in_features
        elif backbone == "resnet50":
            net = tvm.resnet50(weights=None)
            feat_dim = net.fc.in_features
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove the final classification layer — we don't classify during SSL
        net.fc = nn.Identity()
        self.backbone = net
        self.feat_dim = feat_dim

        self.projection_head = ProjectionHead(
            in_dim=feat_dim,
            hidden_dim=hidden_dim,
            out_dim=projection_dim,
        )

    def forward(self, x: torch.Tensor):
        """Return (features, projections). Features are what we use downstream."""
        features = self.backbone(x)
        projections = self.projection_head(features)
        return features, projections

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding extraction — only the backbone, no projection."""
        return self.backbone(x)


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled cross-entropy loss — the core of SimCLR.

    Given a batch of N image pairs (2N views total):
        - Each view has ONE positive pair (the other view of the same image)
        - All other 2N-2 views are negatives
        - We want positives to be close and negatives far in cosine similarity
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2 shape: (N, D) — projections of the two views of the same N images.
        """
        batch_size = z1.shape[0]
        device = z1.device

        # L2-normalize so dot product = cosine similarity
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # Stack all views: shape (2N, D)
        z = torch.cat([z1, z2], dim=0)

        # Pairwise cosine similarities: (2N, 2N)
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarities on the diagonal
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float("-inf"))

        # For each view i in [0, N), its positive is at index i + N, and vice versa
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device),
        ])

        return self.cross_entropy(sim, targets)
