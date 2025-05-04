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