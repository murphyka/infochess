"""GRU model for king location prediction with dual output heads."""

import torch
import torch.nn as nn


class DualHeadGRU(nn.Module):
    """GRU encoder with two output heads.

    first_order_head: predicts opponent king location (softmax probability over squares)
    second_order_head: predicts which squares are visible to the opponent (sigmoid per square)
    """

    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.first_order_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.second_order_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, hidden=None):
        x, hidden = self.rnn(x, hidden)
        return self.first_order_head(x), self.second_order_head(x), hidden


def save_model(model, path, input_dim, output_dim, hidden_dim, num_layers):
    torch.save({
        'state_dict': model.state_dict(),
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
    }, path)


def load_model(path):
    """Load model from path. Returns (model, metadata_dict)."""
    checkpoint = torch.load(path, weights_only=False)
    model = DualHeadGRU(
        checkpoint['input_dim'],
        checkpoint['output_dim'],
        checkpoint['hidden_dim'],
        checkpoint['num_layers'],
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
