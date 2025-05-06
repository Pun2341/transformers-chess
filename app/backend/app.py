# app/backend/app.py
import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class MoveRequest(BaseModel):
    fen: str

app = FastAPI()

# Allow the React dev server (on localhost:3000) to talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] to allow all
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/move")
async def move(req: MoveRequest):
    """
    Dummy Black reply: always play e7e5 if legal; otherwise the first legal move.
    """
    try:
        board = chess.Board(req.fen)
    except Exception as e:
        raise HTTPException(400, detail=f"Invalid FEN: {e}")

    desired = chess.Move.from_uci("e7e5")
    if desired in board.legal_moves:
        reply = desired
    else:
        # fallback to any legal move
        reply = next(iter(board.legal_moves))

    return {"uci": reply.uci()}
