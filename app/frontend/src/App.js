import React, { useState, useCallback } from 'react';
import { Chess } from 'chess.js';
import {Chessboard} from 'react-chessboard';

function calculateBoardWidth() {
  const padding = 20;
  const minDimension = Math.min(window.innerWidth, window.innerHeight);
  return Math.max(minDimension - padding, 300);
}

function ChessGame() {
  const [game, setGame] = useState(new Chess());
  const [fen, setFen] = useState(game.fen());

  function doPlayerMove(sourceSquare, targetSquare) {
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
    if (game.turn() !== 'w') return false
    if (game.isGameOver()) {
      return true;
    }

    if (!doPlayerMove(sourceSquare, targetSquare)) return false;
    
    if (game.isGameOver()) {
      return true;
    }
    doCompMove();

    return true;
  }

  return (<div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
    <div>
      <p>{game.isGameOver() ? 'Game Over' : ''}</p>
    </div>
    <div>
      <Chessboard
        position={fen}
        boardWidth={calculateBoardWidth()}
        onPieceDrop={onDrop}
      />
    </div>
  </div>);
}

export default ChessGame;