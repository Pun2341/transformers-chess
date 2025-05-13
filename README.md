# transformers-chess

### Introduction
This GitHub repository contains our re-implementation of the transformer-based 
chess engine described in the 2024 DeepMind paper, "Amortized Planning with 
Large-Scale Transformers: A Case Study on Chess" by Ruoss et al. Our work 
replicates a key result from the paper using a simplified decoder-only 
transformer to predict win percentages of chess moves, approximating traditional 
search-based engines without explicit planning.

### Chosen Results
We focused on replicating the action-value prediction task, where the model 
outputs win probabilities for each legal move given a board state. This task is 
core to the paper's claim that large transformers can learn planning policies 
directly from data.

### GitHub Contents
* app/: Playable web interface for engine
* checkpoints/: Weights of trained model
* data/: Scripts for downloading dataset
* poster/: Image of final poster
* report/: PDF of final report 
* results/: Puzzle evaluations
* src/: Code for model, preprocessing, evalutation, and testing

### Re-implementation Details
Our model is a decoder-only transformer implemented in PyTorch, trained on a 
subset of the ChessBench dataset with ~1GB of Stockfish-annotated positions. 
Key components include:

* Tokenizing board states (FEN) and moves
* Binning win probabilities into discrete buckets
* Simplified BagReader for reading .bag datasets
* Transformer implementation and training loop
* Evaluation on the puzzle subset to estimate Elo

Due to GPU limitations (2hrs/day on Google Colab), we used a 270K parameter 
model (vs. 270M in the paper), no data sharding, and lightweight heuristics to 
enhance predictions.

### Reproduction Steps
To reproduce our results:

Clone the repo: git clone git@github.com:Pun2341/transformers_chess.git 

To run `transformers-chess`, install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html). Then install the requirements and activate the environment via

```bash
conda create -n chess python=3.12
conda activate chess
pip install -r requirements.txt
```

To run our inference code, enter the following command from the root directory (`transformers-chess/`):

```bash
python -m src.play_chess
```

To run our puzzle evaluation code, enter the following command from the root directory (`transformers-chess/`):

```bash
python -m src.evaluate_puzzles
```

To play against our bot using the web interface, enter the following command from 
the root directory (`transformers-chess/`) using two different terminals:

```bash
uvicorn app.backend.app:app --reload --host 0.0.0.0 --port 8000

# Open a new terminal
npm install --prefix app/frontend  # First time setup
npm start --prefix app/frontend
```

Hardware: GPU highly recommended (tested on Colab with T4)

### Results/Insights
Our model reached ~600 Elo from supervised learning alone.

Adding handcrafted heuristics improved puzzle performance to ~1400 Elo.

Accuracy was highest on easier puzzles (rating < 800) and decreased with 
complexity, consistent with trends from the original paper.

Compared to the paper's 270M-parameter model (achieving 2800 Elo), our results 
highlight how data scale and model size significantly impact generalization.

### Conclusion
This project validated that transformer-based architectures can learn meaningful 
chess strategies without search, even at small scales. Key lessons:

* Compute is a large bottleneck for training transformers from scratch
* Simple rule-based heuristics can enhance low-capacity models.
* Scaling up model size and training data is necessary for full reproduction of 
results.

Future directions include experimenting with deeper architectures, 
continuous-value losses, and hybrid planning mechanisms combining neural and 
symbolic reasoning.

### References
Ruoss, A., Delétang, G., Medapati, S., Grau-Moya, J., Wenliang, L. K., Catt, E., 
Reid, J., Lewis, C. A., Veness, J., & Genewein, T. (2024). Amortized Planning 
with Large-Scale Transformers: A Case Study on Chess. arXiv. 
https://doi.org/10.48550/arXiv.2402.04494Google Scholar+8arXiv+8Hugging Face+8 

### Acknowledgements
This work was completed as part of our course project for CS 4782 SP25. 
We thank our instructor and peers for their feedback and support. Special thanks 
to the authors of the original paper, Ruoss et al.,for releasing their dataset 
and inspiring this exploration into chess engines.
