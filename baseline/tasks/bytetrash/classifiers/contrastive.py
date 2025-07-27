import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ContrastiveByteDataset(Dataset):
    """Dataset for contrastive learning of byte sequences"""

    def __init__(self, strings: List[str], labels: List[bool]):
        self.strings = strings
        self.labels = labels

        # Group samples by label for efficient positive pair sampling
        self.positive_indices = [i for i, label in enumerate(labels) if label]
        self.negative_indices = [i for i, label in enumerate(labels) if not label]

    def __len__(self):
        return len(self.strings)

    def string_to_bytes(self, s: str) -> torch.Tensor:
        """Convert string to byte values, pad/truncate to 128"""
        byte_values = [ord(c) for c in s[:128]]
        if len(byte_values) < 128:
            byte_values.extend([0] * (128 - len(byte_values)))
        return torch.tensor(byte_values, dtype=torch.float32) / 255.0  # Normalize to [0,1]

    def __getitem__(self, idx):
        anchor_string = self.strings[idx]
        anchor_label = self.labels[idx]
        anchor_bytes = self.string_to_bytes(anchor_string)

        # Sample a positive (same class) and negative (different class)
        if anchor_label:
            # Ensure we don't sample the anchor itself
            valid_positive = [i for i in self.positive_indices if i != idx]
            positive_idx = random.choice(valid_positive) if valid_positive else idx
            negative_idx = random.choice(self.negative_indices)
        else:
            valid_positive = [i for i in self.negative_indices if i != idx]
            positive_idx = random.choice(valid_positive) if valid_positive else idx
            negative_idx = random.choice(self.positive_indices)

        positive_bytes = self.string_to_bytes(self.strings[positive_idx])
        negative_bytes = self.string_to_bytes(self.strings[negative_idx])

        return {
            'anchor': anchor_bytes,
            'positive': positive_bytes,
            'negative': negative_bytes,
            'label': torch.tensor(1 if anchor_label else 0, dtype=torch.long)
        }


class ByteStringEncoder(nn.Module):
    """Enhanced encoder for byte sequences"""

    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128, dropout=0.2):
        super().__init__()

        # Use 1D convolutions to capture local patterns in byte sequences
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, 128)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, 128)
        x = self.conv_layers(x)
        x = x.squeeze(-1)  # Remove spatial dimension: (batch_size, 128)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalize


class ContrastiveModel(nn.Module):
    """Full contrastive learning model with triplet loss"""

    def __init__(self, encoder: ByteStringEncoder, margin: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor_emb = self.encoder(anchor)
        positive_emb = self.encoder(positive)
        negative_emb = self.encoder(negative)
        return anchor_emb, positive_emb, negative_emb

    def triplet_loss(self, anchor, positive, negative):
        """Compute triplet loss with better numerical stability"""
        positive_dist = F.pairwise_distance(anchor, positive, p=2)
        negative_dist = F.pairwise_distance(anchor, negative, p=2)

        # Add small epsilon for numerical stability
        loss = F.relu(positive_dist - negative_dist + self.margin)

        # Also return distances for monitoring
        return loss.mean(), positive_dist.mean(), negative_dist.mean()


class ImprovedContrastiveModel(nn.Module):
    """Contrastive model with better loss and training dynamics"""

    def __init__(self, encoder: ByteStringEncoder, margin: float = 0.5, temperature: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.margin = margin
        self.temperature = temperature

    def forward(self, anchor, positive=None, negative=None):
        if positive is None and negative is None:
            # Inference mode
            return self.encoder(anchor)

        anchor_emb = self.encoder(anchor)
        positive_emb = self.encoder(positive)
        negative_emb = self.encoder(negative)
        return anchor_emb, positive_emb, negative_emb

    def triplet_loss(self, anchor, positive, negative):
        """Compute triplet loss with better numerical stability"""
        positive_dist = F.pairwise_distance(anchor, positive, p=2)
        negative_dist = F.pairwise_distance(anchor, negative, p=2)

        # Add small epsilon for numerical stability
        loss = F.relu(positive_dist - negative_dist + self.margin)

        # Also return distances for monitoring
        return loss.mean(), positive_dist.mean(), negative_dist.mean()


class LinearClassifier(nn.Module):
    """Simple linear classifier on top of frozen encoder"""

    def __init__(self, encoder: ByteStringEncoder, input_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)