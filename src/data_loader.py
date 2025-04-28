import torch
from torch.utils.data import Dataset
from tokenizer import tokenize
from utils import Policy, MOVE_TO_ACTION, ACTION_TO_MOVE
from bagz import BagReader
from apache_beam import coders

CODERS = {
    'fen': coders.StrUtf8Coder(),
    'move': coders.StrUtf8Coder(),
    'count': coders.BigIntegerCoder(),
    'win_prob': coders.FloatCoder(),
}

CODERS['action_value'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
    CODERS['win_prob'],
))


def _process_fen(fen: str) -> torch.tensor:
    return tokenize(fen)


def _process_move(move: str) -> torch.tensor:
    return torch.tensor([MOVE_TO_ACTION[move]])


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

    def __len__(self):
        return len(self.bag_reader)

    def __getitem__(self, idx):
        record = self.bag_reader[idx]

        if self.policy == Policy.ACTION_VALUE:
            fen, move, win_prob = CODERS['action_value'].decode(record)
            state = _process_fen(fen)
            action = _process_move(move)
            return_bucket = _process_win_prob(
                win_prob, self._return_buckets_edges)
            sequence = torch.from_numpy(
                torch.concatenate([state, action, return_bucket]))
        else:
            raise ValueError(f"Unknown policy type: {self.policy}")

        return sequence
