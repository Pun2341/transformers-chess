import React, { useState, useCallback } from 'react';
import { Chess } from 'chess.js';
import {Chessboard} from 'react-chessboard';

function ChessGame() {
  const [game, setGame] = useState(new Chess());
  const [fen, setFen] = useState(game.fen());

  function doPlayerMove(sourceSquare, targetSquare) {
    if (game.isGameOver()) return false;

    try {
      game.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q'
      });
      setFen(game.fen());
      return true;
    } catch (e) {
      return false;
    }
  }
  
  const doCompMove = useCallback(async (sourceSquare, targetSquare) => {
    if (game.isGameOver()) return false;

    try {
      const res = await fetch('/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: game.fen() }),
      });
      const { uci } = await res.json();

      game.move(uci);
      setFen(game.fen());          
    } catch (err) {
      console.error('Backend error:', err);
    }

    return true;
  }, [game]);

  function onDrop(sourceSquare, targetSquare) {
    if (!doPlayerMove(sourceSquare, targetSquare)) return false;

    doCompMove();

    return true;
  }

  return (
    <div>
      <Chessboard
        position={fen}
        boardWidth={750}
        onPieceDrop={onDrop}
      />
    </div>
  );
}

export default ChessGame;