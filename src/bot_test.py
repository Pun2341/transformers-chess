from src.engine import Engine, EngineWithHeuristics
from src.transformer import TransformerConfig, TransformerDecoder, PositionalEncodings, Predictor
from src.utils import MOVE_TO_ACTION
from src.tokenizer import SEQUENCE_LENGTH
import torch
import chess


def get_predictor(model_path, transformer_config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerDecoder(transformer_config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return Predictor(model)


def play_game(predictor_white, predictor_black, max_moves=100):
    board = chess.Board()
    # engine_white = Engine(predictor_white, starting_fen=board.fen())
    # engine_black = Engine(predictor_black, starting_fen=board.fen())
    engine_white = EngineWithHeuristics(predictor_white, starting_fen=board.fen())
    engine_black = EngineWithHeuristics(predictor_black, starting_fen=board.fen())

    move_count = 0
    print("Starting new game...\n")

    while not board.is_game_over() and move_count < max_moves:
        engine = engine_white if board.turn == chess.WHITE else engine_black
        move = engine.get_best_move()

        if move is None or move not in board.legal_moves:
            print("Illegal or None move, skipping turn.")
            break

        move_san = board.san(move)
        print(f"{move_count + 1}. {'White' if board.turn == chess.WHITE else 'Black'} plays: {move_san}")
        board.push(move)

        # Sync both engine boards
        engine_white.board = board.copy()
        engine_black.board = board.copy()

        move_count += 1

    print("\nFinal Board Position:")
    print(board)

    result = board.result() if board.is_game_over() else "1/2-1/2 (draw by move cap)"
    print("\nGame Result:", result)
    return result


def main():
    model_path_1 = "src/checkpoint_epoch4_20250506_053148.pt"
    model_path_2 = "src/checkpoint_epoch5_step5544.pt"

    config1 = TransformerConfig(
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
    
    config2 = TransformerConfig(
        vocab_size=len(MOVE_TO_ACTION),
        output_size=128,
        pos_encodings=PositionalEncodings.SINUSOID,
        max_sequence_length=SEQUENCE_LENGTH + 2,
        num_heads=4,
        num_layers=4,
        embedding_dim=64,
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )

    predictor1 = get_predictor(model_path_1, config1)
    predictor2 = get_predictor(model_path_2, config2)

    play_game(predictor1, predictor2)


if __name__ == "__main__":
    main()
