from src.engine import Engine, EngineWithHeuristics
from src.transformer import TransformerConfig, PositionalEncodings, TransformerDecoder, Predictor
from src.utils import MOVE_TO_ACTION
from src.tokenizer import SEQUENCE_LENGTH
import torch

num_return_buckets = 128


def get_predictor(model_path, transformer_config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerDecoder(transformer_config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    predictor = Predictor(model)
    return predictor


if __name__ == "__main__":
    model_path = "src/checkpoint_epoch2_20250506_040633.pt"
    transformer_config = TransformerConfig(
        vocab_size=len(MOVE_TO_ACTION),
        output_size=num_return_buckets,
        pos_encodings=PositionalEncodings.SINUSOID,
        max_sequence_length=SEQUENCE_LENGTH + 2,
        num_heads=4,
        num_layers=2,
        embedding_dim=64,
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )

    predictor = get_predictor(model_path, transformer_config)
    starting_board = None
    # chess_engine = Engine(predictor, starting_board)
    chess_engine = EngineWithHeuristics(predictor, starting_board)

    while True:
        color = input("Black or White? ")
        if color.lower() == "black":
            move = chess_engine.computer_play()
            break
        elif color.lower() == "white":
            break
    while True:
        move = input("What move would you like to play: ")
        if move.lower() == "quit":
            break
        if move.lower() == "board":
            print(chess_engine.board.fen())
            continue
        move = chess_engine.human_play(move)
        if move is None:
            print("Illegal move, try again.")
        chess_engine.computer_play()
