# app/inference/model.py

import numpy as np
import chess

from transformer import Config, make_predictor, wrap_predict_fn
from engine import ActionValueEngine

# 1) Load once at import time
cfg       = Config.from_name("136M")            # or your checkpoint alias
predictor, params = make_predictor(cfg)
predict_fn = wrap_predict_fn(predictor, params, batch_size=cfg.batch_size)
return_buckets = np.array(cfg.return_buckets, dtype=np.float32)

engine = ActionValueEngine(
    return_buckets_values=return_buckets,
    predict_fn=predict_fn,
    temperature=0.5,
)

def predict_move(fen: str) -> str:
    """
    Given a FEN string, returns the best move in UCI format.
    """
    board = chess.Board(fen)
    move  = engine.play(board)
    return move.uci()