import torch
import numpy as np
from torch.utils.data import Dataset
from src.tokenizer import process_fen, process_move, CODERS
from src.utils import Policy
from src.bagz import BagReader


def _process_win_prob(
    win_prob: float,
    bins_edges: torch.tensor,
) -> torch.tensor:
    returns = torch.tensor([win_prob])
    if returns.dim() != 1:
        raise ValueError(f'Returns should be 1D, got {returns.dim()}D.')

    if bins_edges.dim() != 1:
        raise ValueError(f'Bin edges should be 1D, got {bins_edges.dim()}D.')

    return torch.searchsorted(bins_edges, returns, right=False)


def _get_uniform_buckets_edges_values(num_buckets: int) -> tuple[torch.Tensor, torch.Tensor]:
    full_linspace = torch.linspace(0.0, 1.0, steps=num_buckets + 1)

    edges = full_linspace[1:-1]
    values = (full_linspace[:-1] + full_linspace[1:]) / 2

    return edges, values


class ChessDataset(Dataset):
    def __init__(self, bag_path, policy=Policy.ACTION_VALUE, num_return_buckets=128):
        self.bag_reader = BagReader(bag_path)
        self.policy = policy
        self.num_return_buckets = num_return_buckets
        self._return_buckets_edges, _ = _get_uniform_buckets_edges_values(
            num_return_buckets,
        )
        self._sequence_length = tokenizer.SEQUENCE_LENGTH + \
            2 if policy == Policy.ACTION_VALUE else tokenizer.SEQUENCE_LENGTH + 1
        self._loss_mask = np.full(
            shape=(self._sequence_length,),
            fill_value=True,
            dtype=bool,
        )
        self._loss_mask[-1] = False

    def __len__(self):
        return len(self.bag_reader)

    def __getitem__(self, idx):
        record = self.bag_reader[idx]

        if self.policy == Policy.ACTION_VALUE:
            fen, move, win_prob = CODERS['action_value'].decode(record)
            if len(fen.split(' ')) != 6:
                raise ValueError(
                    f"Invalid FEN string: {fen} at index {idx}")

            state = process_fen(fen)
            action = process_move(move)
            return_bucket = _process_win_prob(
                win_prob, self._return_buckets_edges)
            sequence = torch.concatenate([state, action, return_bucket])
        elif self.policy == Policy.STATE_VALUE:
            fen, win_prob = CODERS['state_value'].decode(record)
            state = process_fen(fen)
            return_bucket = _process_win_prob(
                win_prob, self._return_buckets_edges)
            sequence = torch.concatenate([state, return_bucket])
        elif self.policy == Policy.BEHAVIORAL_CLONING:
            fen, move = CODERS['behavioral_cloning'].decode(record)
            state = process_fen(fen)
            action = process_move(move)
            sequence = torch.concatenate([state, action])
        else:
            raise ValueError(f"Unknown policy type: {self.policy}")
        return sequence, self._loss_mask
