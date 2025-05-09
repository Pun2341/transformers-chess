import pandas as pd
import chess
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

from src.engine import Engine, EngineWithHeuristics
from src.transformer import TransformerConfig, TransformerDecoder, PositionalEncodings, Predictor
from src.utils import MOVE_TO_ACTION
from src.tokenizer import SEQUENCE_LENGTH


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


def get_rating_bin(rating):
    """Group puzzle ratings into bins (≤800, 800–999, ..., 2000–2199, etc.)."""
    if rating <= 800:
        return "≤800"
    else:
        low = int((rating - 1) // 200 * 200)
        high = low + 199
        return f"{low}-{high}"


def estimate_elo_from_score(target_score, bin_scores_map):
        rating_bins = sorted(bin_scores_map.items(), key=lambda x: int(x[0].split('-')[0].replace('≤', '0')))
        bin_ranges = [int(b[0].split('-')[0].replace('≤', '0')) for b in rating_bins]
        avg_scores = [sum(scores) / len(scores) for _, scores in rating_bins]

        for i in range(1, len(avg_scores)):
            if avg_scores[i-1] >= target_score >= avg_scores[i] or avg_scores[i-1] <= target_score <= avg_scores[i]:
                x0, x1 = avg_scores[i-1], avg_scores[i]
                y0, y1 = bin_ranges[i-1], bin_ranges[i]
                if x0 == x1:
                    return y0
                estimated_elo = y0 + (target_score - x0) * (y1 - y0) / (x1 - x0)
                return round(estimated_elo)

        if target_score <= avg_scores[-1]:
            return bin_ranges[-1]
        if target_score >= avg_scores[0]:
            return bin_ranges[0]
        return None


def evaluate_puzzles(csv_path="src/puzzles.csv", max_puzzles=None, engine=None):
    df = pd.read_csv(csv_path)
    if max_puzzles:
        df = df.head(max_puzzles)

    if engine is None:
        predictor = load_model()
        engine = EngineWithHeuristics(predictor)

    bin_scores = defaultdict(list)
    detailed_results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        fen = row["FEN"]
        rating = row["Rating"]
        solution_moves = row["Moves"].split()
        if not solution_moves:
            continue

        board = chess.Board(fen)
        engine.board = board

        total_moves = 0  # number of model moves
        correct_moves = 0
        predicted_moves = []

        try:
            for expected_uci in solution_moves:
                model_turn = engine.board.turn

                if model_turn:
                    predicted_move = engine.get_best_move()
                    predicted_uci = predicted_move.uci()
                    predicted_moves.append(predicted_uci)

                    if predicted_uci == expected_uci:
                        correct_moves += 1

                    engine.board.push(predicted_move)
                    total_moves += 1
                else:
                    # Opponent's move: use expected move to keep game flow correct
                    move = chess.Move.from_uci(expected_uci)
                    predicted_moves.append(expected_uci + "*")
                    engine.board.push(move)

        except Exception as e:
            continue

        score = correct_moves / total_moves if total_moves > 0 else 0.0
        bin_key = get_rating_bin(rating)
        bin_scores[bin_key].append(score)

        detailed_results.append({
            "PuzzleId": row["PuzzleId"],
            "Rating": rating,
            "FEN": fen,
            "Expected": " ".join(solution_moves),
            "Predicted": " ".join(predicted_moves),
            "Score": score
        })

    # Save results
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv("puzzle_eval_results.csv", index=False)

    # Plot average accuracy per bin
    bins_sorted = sorted(bin_scores.keys(), key=lambda x: int(x.split('-')[0].replace('≤', '0')))
    avg_scores = [sum(bin_scores[b]) / len(bin_scores[b]) for b in bins_sorted]

    plt.figure(figsize=(10, 6))
    plt.bar(bins_sorted, avg_scores, color='steelblue')
    plt.xlabel("Puzzle Rating Range")
    plt.ylabel("Average Accuracy Score")
    plt.title("Transformer Puzzle Accuracy by Rating")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("puzzle_accuracy_by_rating.png")
    plt.close()

    print(f"\nEvaluated {len(results_df)} puzzles")
    print("Results saved to 'puzzle_eval_results.csv' and plot saved to 'puzzle_accuracy_by_rating.png'")
    
    
    # Compute the weighted mean and median score over all puzzles
    scores = results_df["Score"]
    ratings = results_df["Rating"]

    weighted_mean_score = (scores * ratings).sum() / ratings.sum()

    # sorted_df = results_df.sort_values("Rating")
    # cumulative_weight = sorted_df["Rating"].cumsum()
    # half_weight = sorted_df["Rating"].sum() / 2
    # weighted_median_score = sorted_df.loc[cumulative_weight >= half_weight, "Score"].iloc[0]

    estimated_weighted_mean_elo = estimate_elo_from_score(weighted_mean_score, bin_scores)
    # estimated_weighted_median_elo = estimate_elo_from_score(weighted_median_score, bin_scores)
    
    with open("puzzle_eval_summary.txt", "w") as f:
        f.write(f"Puzzle Evaluation Summary\n")
        f.write(f"{'-'*30}\n")
        f.write(f"Total puzzles evaluated: {len(results_df)}\n")
        f.write(f"Weighted mean score: {weighted_mean_score:.3f}\n")
        # f.write(f"Weighted median score: {weighted_median_score:.3f}\n")
        f.write(f"Estimated ELO (weighted mean-based): {estimated_weighted_mean_elo}\n")
        # f.write(f"Estimated ELO (weighted median-based): {estimated_weighted_median_elo}\n")
        f.write(f"\nScore distribution by rating bin:\n")
        for b in bins_sorted:
            avg = sum(bin_scores[b]) / len(bin_scores[b])
            f.write(f"  {b}: avg score = {avg:.3f}, puzzles = {len(bin_scores[b])}\n")
            
    return results_df, bin_scores


if __name__ == "__main__":
    evaluate_puzzles("src/puzzles.csv", max_puzzles=1000)
