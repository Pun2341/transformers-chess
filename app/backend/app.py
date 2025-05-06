from flask import Flask, request, jsonify
from flask_cors import CORS
from engine import get_best_move  # Replace with your model's function

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/move', methods=['POST'])
def move():
    data = request.get_json()
    fen = data.get('fen')
    if not fen:
        return jsonify({'error': 'FEN not provided'}), 400
    move = get_best_move(fen)  # Implement this function in your model
    return jsonify({'move': move})


if __name__ == '__main__':
    app.run(debug=True)
