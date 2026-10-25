from __future__ import annotations

import torch
import torch.nn as nn


class NeighborhoodGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 96, dropout: float = 0.2):
        super().__init__()
        self.self_encoder = _encoder(in_dim, hidden_dim, dropout)

        self.first_neighbor_encoder = _encoder(in_dim, hidden_dim, dropout)
        self.first_mean_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.first_norm = nn.LayerNorm(hidden_dim)

        self.second_neighbor_encoder = _encoder(in_dim, hidden_dim, dropout)
        self.second_mean_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.second_norm = nn.LayerNorm(hidden_dim)

        self.projector = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.self_decoder = nn.Linear(hidden_dim, in_dim, bias=False)
        self.first_environment_decoder = nn.Linear(hidden_dim, in_dim, bias=False)
        self.second_environment_decoder = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x, first_weights, second_weights):
        h_self = self.self_encoder(x)

        first_states = self.first_neighbor_encoder(x)
        mean1 = torch.sparse.mm(first_weights, first_states)
        h_first = self.first_norm(self.first_mean_encoder(mean1))

        second_states = self.second_neighbor_encoder(x)
        mean2 = torch.sparse.mm(second_weights, second_states)
        h_second = self.second_norm(self.second_mean_encoder(mean2))

        return self.projector(torch.cat((h_self, h_first, h_second), dim=1))

    def decode_self_state(self, embedding):
        return self.self_decoder(embedding)

    def decode_first_environment(self, embedding):
        return self.first_environment_decoder(embedding)

    def decode_second_environment(self, embedding):
        return self.second_environment_decoder(embedding)


def _encoder(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout),
    )
