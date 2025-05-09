// GameOverModal.js
import React from 'react';

export default function GameOverModal({ open, result, onRestart }) {
  if (!open) return null;

  return (
    <div style={styles.overlay}>
      <div style={styles.modal}>
        <h2>Game Over</h2>
        <p>{result}</p>
        <button onClick={onRestart} style={styles.button}>Play Again</button>
      </div>
    </div>
  );
}

const styles = {
  overlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    backgroundColor: 'rgba(0,0,0,0.5)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  modal: {
    backgroundColor: '#fff',
    padding: '2rem',
    borderRadius: '8px',
    textAlign: 'center',
    minWidth: '300px',
  },
  button: {
    marginTop: '1rem',
    padding: '0.5rem 1rem',
    fontSize: '1rem',
  },
};
