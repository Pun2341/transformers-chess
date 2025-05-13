# transformers-chess
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

To play against our bot, enter the following command from the root directory using
two different terminals

```bash
uvicorn app.backend.app:app --reload --host 0.0.0.0 --port 8000

# Open a new terminal
npm start --prefix app/frontend
```