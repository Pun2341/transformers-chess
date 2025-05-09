from src.engine import Engine
from src.transformer import TransformerConfig, PositionalEncodings, TransformerDecoder, Predictor
from src.utils import MOVE_TO_ACTION
from src.tokenizer import SEQUENCE_LENGTH
import torch


class Bot:
    def __init__(self, model_path):
        transformer_config = TransformerConfig(
            vocab_size=len(MOVE_TO_ACTION),
            output_size=128,
            pos_encodings=PositionalEncodings.SINUSOID,
            max_sequence_length=SEQUENCE_LENGTH + 2,
            num_heads=4,
            num_layers=2,
            embedding_dim=64,
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = TransformerDecoder(transformer_config)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])

        predictor = Predictor(model)
        self.engine = Engine(predictor)

    def nextMove(self, board):
        try:
            return self.engine.get_best_move_from_fen(board)
        except Exception as e:
            print(e)
            return None
