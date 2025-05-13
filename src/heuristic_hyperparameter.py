from src.engine import EngineWithHeuristics
from src.evaluate_puzzles import evaluate_puzzles, estimate_elo_from_score
import random
from src.transformer import TransformerConfig, TransformerDecoder, PositionalEncodings, Predictor
from src.utils import MOVE_TO_ACTION
from src.tokenizer import SEQUENCE_LENGTH
import torch


def load_model(checkpoint_path="src/checkpoint_epoch4_20250506_053148.pt"):
    config = TransformerConfig(
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
    model = TransformerDecoder(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return Predictor(model)


def random_weight_config():
    CONST = 5
    return {
        'material':     random.uniform(0, 1) * CONST,
        'center':       random.uniform(0, 1) * CONST,
        'check':        random.uniform(0, 1) * CONST,
        'capture':      random.uniform(0, 1) * CONST,
        'development':  random.uniform(0, 1) * CONST,
        'blunder':      random.uniform(0, 1) * CONST,
        'trade':        random.uniform(0, 1) * CONST, 
        'hanging':      random.uniform(0, 1) * CONST
    }


def run_search(trials=50, csv_path="src/puzzles.csv"):
    predictor = load_model()
    best_config = None
    best_score = float('-inf')

    for i in range(trials):
        weights = random_weight_config()
        print(f"\nTrial {i+1}/{trials}: Testing weights -> {weights}")
        
        # Override the engine temporarily
        engine = EngineWithHeuristics(predictor, weights=weights)

        # Evaluate
        results_df, bin_scores = evaluate_puzzles(csv_path=csv_path, engine=engine, max_puzzles=200)
        mean_score = results_df["Score"].mean()
        estimated_elo = estimate_elo_from_score(mean_score, bin_scores)

        print(f"Mean score: {mean_score:.3f} | Estimated ELO: {estimated_elo}")

        if estimated_elo > best_score:
            best_score = estimated_elo
            best_config = weights

    print("\nBest weights found:")
    print(best_config)
    print(f"Estimated ELO: {best_score}")


if __name__ == "__main__":
    run_search()