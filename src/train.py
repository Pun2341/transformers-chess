import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from transformers_chess.src import config as config_lib
from transformers_chess.src import datasets
from transformers_chess.src import transformer, TransformerConfig


def train(config: config_lib.Config, rng_seed: int = 0):
    # Set random seeds
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)

    # Load dataset
    train_ds = datasets.load_dataset(config.train_dataset)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True)

    # Create model
    model = transformer.TransformerDecoder(config)
    model.train()
    model.to(config.device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Optionally resume from checkpoint
    start_epoch = 0
    checkpoint_path = os.path.join(config.checkpoint_dir, "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        total_loss = 0.0
        for batch in train_loader:
            x, y = batch
            x, y = x.to(config.device), y.to(config.device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        # Save checkpoint
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, checkpoint_path)

    return model


if __name__ == "__main__":
    config = TransformerConfig()
    model = train(config)
