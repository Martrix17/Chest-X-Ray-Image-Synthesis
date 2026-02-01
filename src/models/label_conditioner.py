"""
MultiLabelConditioner class for label embeddings.
"""

import torch
import torch.nn as nn


class MultiLabelConditioner(nn.Module):
    """Converts multilabel one-hot vectors to embeddings for diffusion conditioning."""

    def __init__(
        self,
        num_classes: int = 11,
        embed_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            num_classes: Number of classes to embed.
            embed_dim: Number of embedding dimensions.
            dropout: Dropout fraction.
        """
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.class_embeddings = nn.Embedding(num_classes, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: Tensor of shape [B, num_classes], binary (0/1)

        Returns:
            encoder_hidden_states: [B, num_classes, embed_dim]
        """
        device = labels.device
        class_indices = torch.arange(self.num_classes, device=device)

        tokens = self.class_embeddings(class_indices)
        tokens = tokens.unsqueeze(0).expand(labels.shape[0], -1, -1)
        tokens = tokens * labels.unsqueeze(-1)
        tokens = self.dropout(tokens)
        return tokens
