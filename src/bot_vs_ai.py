import os
import time
import chess
import torch
import berserk
from dotenv import load_dotenv

from src.engine import Engine, EngineWithHeuristics
from src.transformer import TransformerConfig, TransformerDecoder, PositionalEncodings, Predictor
from src.utils import MOVE_TO_ACTION
from src.tokenizer import SEQUENCE_LENGTH


# Load Lichess API token from .env file
load_dotenv()
token = os.getenv("LICHESS_TOKEN")
if not token:
    raise RuntimeError("LICHESS_TOKEN not set in environment.")

# Connect to Lichess
session = berserk.TokenSession(token)
client = berserk.Client(session=session)

# Confirm bot identity
account = client.account.get()
print(f"Logged in as: {account['username']} (Bot ID: {account['id']})")

# Load Transformer model
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
checkpoint = torch.load("src/checkpoint_epoch4_20250506_053148.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state"])
model.eval()

predictor = Predictor(model)
engine = Engine(predictor)
# engine = EngineWithHeuristics(predictor)


# Play the game
def play_game(game_id):
    print(f"Game started! Game ID: {game_id}")
    print(f"Watch here: https://lichess.org/{game_id}")
    
    seen_moves = []

    for event in client.bots.stream_game_state(game_id):
        if event["type"] == "gameFull":
            seen_moves.clear()
            fen = event.get("initialFen", "")
            if fen == "startpos" or not fen:
                engine.board = chess.Board()
            else:
                engine.board = chess.Board(fen)
            my_color = chess.WHITE if event["white"]["id"] == account["id"] else chess.BLACK
            if my_color == chess.WHITE:
                move = engine.get_best_move()
                engine.board.push(move)
                seen_moves.append(move.uci())
                client.bots.make_move(game_id, move.uci())
                
        elif event["type"] == "gameState":
            moves = event.get("moves", "").split()
            new_moves = moves[len(seen_moves):]
            for uci in new_moves:
                try:
                    engine.board.push_uci(uci)
                    seen_moves.append(uci)
                except Exception as e:
                    print(f"[gameState] Error pushing move {uci}: {e}")

            if engine.board.turn == my_color and not engine.board.is_game_over():
                move = engine.get_best_move()
                if move:
                    engine.board.push(move)
                    seen_moves.append(move.uci())
                    client.bots.make_move(game_id, move.uci())
                
        elif event["type"] == "gameFinish":
            print("\nGame over!")
            print("Final board:")
            print(engine.board)
            result = event.get("winner", "draw")
            print("Result:", result)
            break
        
        else:
            print(f"Unhandled Event Type: {event['type']}")


if __name__ == "__main__":
    print("Listening for challenges...")
    for event in client.bots.stream_incoming_events():
        print("Received event:", event)
        if event["type"] == "gameStart":
            game_id = event["game"]["id"]
            play_game(game_id)
            break
        else:
            print(f"Unhandled Event Type: {event['type']}")
