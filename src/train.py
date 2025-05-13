import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from dataclasses import dataclass
import datetime
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader, ConcatDataset
from src.transformer import TransformerDecoder, TransformerConfig, PositionalEncodings, initialize_weights
from src.data_loader import ChessDataset
from src.utils import MOVE_TO_ACTION
from src.tokenizer import SEQUENCE_LENGTH


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 10
    checkpoint_dir: str = "ckpts"
    resume: bool = True
    rng_seed: int = 0


def make_loss_fn():
    def loss_fn(log_probs: torch.Tensor, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = sequences.shape
        device = log_probs.device

        # Flatten inputs
        log_probs = log_probs.view(batch_size * seq_len, -1)
        sequences_flat = sequences.view(-1)
        mask_flat = mask.view(-1)

        # Only keep valid positions (non-masked)
        valid_positions = ~mask_flat.bool()
        log_probs_valid = log_probs[valid_positions]
        sequences_valid = sequences_flat[valid_positions].to(device)

        # Index into correct log-probs for each true token
        true_log_probs = log_probs_valid[
            torch.arange(len(sequences_valid), device=device),
            sequences_valid
        ]

        # Map each token back to its batch index
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, seq_len).reshape(-1)
        batch_indices = batch_indices[valid_positions]

        # Accumulate log probs by batch
        marginal_log_probs = torch.zeros(batch_size, device=device)
        marginal_log_probs = marginal_log_probs.index_add(0, batch_indices, true_log_probs)

        # Count valid tokens per batch item
        seq_lengths = (~mask).int().sum(dim=1).clamp(min=1).to(device)

        # Normalize and compute loss
        loss = -torch.mean(marginal_log_probs / seq_lengths)
        return loss

    return loss_fn


def update_ema(model, ema_model, decay=0.99):
    """Updates EMA model parameters."""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1-decay)


def evaluate(model, dataloader, loss_fn, device):
    """Evaluates model using given dataloader and loss function."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, mask in dataloader:
            x, mask = x.to(device), mask.to(device)
            logits = model(x)
            loss = loss_fn(logits, x, mask)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"[Evaluation] Average loss: {avg_loss:.4f}")
    model.train()
    return avg_loss


def train(
    predictor_config: TransformerConfig,
    train_config: TrainConfig,
    train_dataset: ChessDataset,
    eval_dataset: ChessDataset = None,
    init_weights: bool = True
):
    torch.manual_seed(train_config.rng_seed)
    np.random.seed(train_config.rng_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=train_config.batch_size) if eval_dataset else None

    # Model + EMA
    model = TransformerDecoder(predictor_config).to(device)
    # ema_model = deepcopy(model)
    if train_config.resume and init_weights:
        initialize_weights(model)

    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)
    loss_fn = make_loss_fn()

    start_epoch = 0
    latest_ckpt = os.path.join(train_config.checkpoint_dir, "checkpoint_latest.pt")

    # Resume if applicable
    if train_config.resume and os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt)
        model.load_state_dict(checkpoint["model_state"])
        # ema_model.load_state_dict(checkpoint["ema_model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed training from checkpoint at epoch {start_epoch}")

    os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    step_losses = []
    eval_step_losses = []
    step_numbers = []
    global_step = 0
    
    for epoch in range(start_epoch, train_config.num_epochs):
        total_loss = 0.0
        model.train()

        for i, (x, mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            x, mask = x.to(device), mask.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, x, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # update_ema(model, ema_model, decay=0.99)
            total_loss += loss.item()
            global_step += 1
            
            if global_step % 20000 == 0:
                avg_step_loss = total_loss / (i + 1)
                step_losses.append(avg_step_loss)
                step_numbers.append(global_step)
                print(f"[Step {global_step} in Epoch {epoch+1}], Loss: {avg_step_loss:.4f}")
                
                if eval_loader:
                    eval_loss = evaluate(model, eval_loader, loss_fn, device)
                    eval_step_losses.append(eval_loss)
                else:
                    eval_step_losses.append(None)

                checkpoint_path = os.path.join(
                    train_config.checkpoint_dir,
                    f"checkpoint_epoch{epoch+1}_step{i+1}.pt"
                )
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict()
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

        avg_epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_epoch_loss:.4f}")

        # Evaluate if eval dataset provided
        if eval_loader:
            evaluate(model, eval_loader, loss_fn, device)

        # Save checkpoint with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch{epoch+1}_{timestamp}.pt"
        checkpoint_path = os.path.join(train_config.checkpoint_dir, checkpoint_name)

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            # "ema_model_state": ema_model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
        
        plt.figure(figsize=(8, 5))
        plt.plot(step_numbers, step_losses, marker='o', label='Train Loss')
        if any(val is not None for val in eval_step_losses):
            plt.plot(step_numbers, eval_step_losses, marker='x', label='Eval Loss')

        plt.title("Loss Over Steps")
        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(train_config.checkpoint_dir, f"loss_epoch{epoch+1}.png")
        plt.savefig(plot_path)
        plt.close()

    print("Finished Training!")
    return model


if __name__ == "main":
    N_BUCKETS = 128

    bag_paths = [
        "train/action_value-00009-of-02148_data.bag",  # 146.7 MB
        "train/action_value-00008-of-02148_data.bag",  #  15.4 MB
        "train/action_value-00007-of-02148_data.bag",  # 361.4 MB
        "train/action_value-00006-of-02148_data.bag",  #  19.7 MB
        "train/action_value-00005-of-02148_data.bag",  #  1.23 GB
        "train/action_value-00004-of-02148_data.bag",  # 208.7 MB
        "train/action_value-00003-of-02148_data.bag",  # 437.2 MB
        "train/action_value-00002-of-02148_data.bag",  #   946 MB
        "train/action_value-00001-of-02148_data.bag",  #  1.58 GB
        "train/action_value-00000-of-02148_data.bag",  #  1.25 GB
    ]

    datasets = [ChessDataset(path, num_return_buckets=N_BUCKETS) for path in bag_paths]
    combined_dataset = ConcatDataset(datasets)
    
    eval_dataset = ChessDataset("test/action_value_data.bag", num_return_buckets=N_BUCKETS)
    
    transformer_config = TransformerConfig(
        vocab_size=len(MOVE_TO_ACTION),
        output_size=N_BUCKETS,
        pos_encodings=PositionalEncodings.SINUSOID,
        max_sequence_length=SEQUENCE_LENGTH + 2,
        num_heads=4,
        num_layers=4,
        embedding_dim=64,
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )
    
    tmp_model = TransformerDecoder(transformer_config)
    total_params = sum(p.numel() for p in tmp_model.parameters())

    train_config = TrainConfig(
        batch_size=64,
        learning_rate=1e-4,
        num_epochs=10,
        checkpoint_dir=f"checkpoints_{total_params}",
        resume=False,
        rng_seed=0,
    )
    
    model = train(transformer_config, 
                  train_config, 
                  train_dataset=combined_dataset, 
                  eval_dataset=eval_dataset)