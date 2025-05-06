import React, { useState } from 'react';
import { Chess } from 'chess.js';
import {Chessboard} from 'react-chessboard';

function ChessGame() {
  const [game, setGame] = useState(new Chess());
  const [fen, setFen] = useState(game.fen());

  function onPieceDrop(sourceSquare, targetSquare) {
    try {
      game.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q' // always promote to a queen for example simplicity
      });
      setFen(game.fen());
      return true;
    } catch (e) {
      return false;
    }
  }

  return (
    <div>
      <Chessboard
        position={fen}
        boardWidth={750}
        onPieceDrop={onPieceDrop}
      />
    </div>
  );
}

export default ChessGame;