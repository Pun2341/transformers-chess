# app/backend/app.py
# uvicorn app:app --reload --host 0.0.0.0 --port 8000
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bot import Bot


class MoveRequest(BaseModel):
    fen: str


PATH = "../../checkpoints/Epoch4.pt"

app = FastAPI()
bot = Bot(PATH)

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
        move = bot.nextMove(req.fen)
        return {"uci": move.uci()}
    except Exception as e:
        raise HTTPException(400, detail=f"Invalid FEN: {e}")
