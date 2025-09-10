from nacasr.layers import compute_input_lengths
from nacasr.utils import prepare_ctc_targets, greedy_decoder, save_model
import evaluate
import torch
import torch.nn.functional as F
from torch.nn import CTCLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import time
import datetime
import logging
from typing import Tuple, Dict, Any
import wandb


def training_step(
    model, train_dataloader, optimizer, scheduler, device, loss_fn, logger
) -> float:
    """
    Perform one training epoch

    Args:
        model: Model to train
        train_dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use
        loss_fn: Loss function
        logger: Logger instance

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    logger.info("Starting training step...")
    start_time = time.time()

    for batch_idx, batch in enumerate(train_dataloader):
        try:
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            # Compute input lengths
            input_lengths = compute_input_lengths(input_values)

            # Ensure input is in correct format (batch, seq_len, features)
            if len(input_values.shape) == 3 and input_values.shape[-1] != 256:
                input_values = input_values.transpose(1, 2)

            # Forward pass
            outputs = model(input_values)

            # Prepare targets for CTC
            targets, target_lengths = prepare_ctc_targets(labels, device)

            # Prepare for CTC loss: (T, B, C)
            log_probs = F.log_softmax(outputs, dim=-1).transpose(0, 1)

            # CTC Loss
            loss = loss_fn(log_probs, targets, input_lengths, target_lengths)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:  # Log every 10 batches
                logger.debug(
                    f"Batch {batch_idx}/{len(train_dataloader)}: Loss = {loss.item():.4f}"
                )

        except Exception as e:
            logger.error(f"Error in training batch {batch_idx}: {str(e)}")
            continue

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    elapsed_time = time.time() - start_time

    logger.info(
        f"Training step completed. Average loss: {avg_loss:.4f}, "
        f"Time: {datetime.timedelta(seconds=int(elapsed_time))}"
    )

    return avg_loss


def eval_step(
    model, device, test_loader, criterion, text_transform, logger
) -> Tuple[float, float]:
    """
    Evaluate model on test set

    Args:
        model: Model to evaluate
        device: Device to use
        test_loader: Test data loader
        criterion: Loss function
        text_transform: Text transformation object
        logger: Logger instance

    Returns:
        Tuple of (average_loss, average_wer)
    """
    model.eval()
    test_loss = 0.0
    test_wer = []
    num_batches = 0

    logger.info("Starting evaluation...")
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)

                # Compute input lengths
                input_lengths = compute_input_lengths(input_values)

                # Ensure input is in correct format
                if len(input_values.shape) == 3 and input_values.shape[-1] != 256:
                    input_values = input_values.transpose(1, 2)

                # Forward pass
                outputs = model(input_values, input_lengths)

                # Prepare targets for CTC
                targets, target_lengths = prepare_ctc_targets(labels, device)

                # Prepare for CTC loss
                log_probs = F.log_softmax(outputs, dim=-1).transpose(0, 1)

                # Compute loss
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                test_loss += loss.item()
                num_batches += 1

                # Decode predictions for WER calculation
                decoded_preds, decoded_targets = greedy_decoder(
                    outputs, labels, text_transform
                )

                # Compute WER
                try:
                    wer_metric = evaluate.load("wer")
                    wer = wer_metric.compute(
                        predictions=decoded_preds, references=decoded_targets
                    )
                    test_wer.append(wer)
                except Exception as e:
                    logger.warning(
                        f"WER computation failed for batch {batch_idx}: {str(e)}"
                    )
                    continue

            except Exception as e:
                logger.error(f"Error in evaluation batch {batch_idx}: {str(e)}")
                continue

    avg_loss = test_loss / num_batches if num_batches > 0 else float("inf")
    avg_wer = sum(test_wer) / len(test_wer) if test_wer else float("inf")
    elapsed_time = time.time() - start_time

    logger.info(
        f"Evaluation completed. Average loss: {avg_loss:.4f}, "
        f"Average WER: {avg_wer:.4f}, "
        f"Time: {datetime.timedelta(seconds=int(elapsed_time))}"
    )

    return avg_loss, avg_wer


def train_full_pipeline(
    model,
    train_dataloader,
    eval_dataloader,
    device,
    text_transform,
    epochs: int = 10,
    lr: float = 1e-3,
    output_dir: str = "./nacasr_model",
    logger: logging.Logger = None,
    to_report=True,
) -> Dict[str, Any]:
    """
    Complete training pipeline

    Args:
        model: Model to train
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
        device: Device to use
        text_transform: Text transformation object
        epochs: Number of epochs
        lr: Learning rate
        output_dir: Output directory for saving models
        logger: Logger instance

    Returns:
        Training statistics dictionary
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Move model to device
    model.to(device)

    # Setup training components
    loss_fn = CTCLoss(blank=0, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    total_steps = len(train_dataloader) * epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="linear",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    logger.info(f"Starting training pipeline...")
    logger.info(f"Device: {device}")
    logger.info(f"Total epochs: {epochs}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training statistics
    training_stats = {
        "epochs": [],
        "train_losses": [],
        "eval_losses": [],
        "eval_wers": [],
        "best_wer": float("inf"),
        "best_epoch": 0,
    }

    best_wer = float("inf")

    for epoch in range(epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'='*50}")

        train_loss = training_step(
            model, train_dataloader, optimizer, scheduler, device, loss_fn, logger
        )

        # Evaluation step
        eval_loss, eval_wer = eval_step(
            model, device, eval_dataloader, loss_fn, text_transform, logger
        )

        logger.info(f"Epoch {epoch + 1} Results:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Eval Loss: {eval_loss:.4f}")
        logger.info(f"  Eval WER: {eval_wer:.4f}")

        # Update statistics
        training_stats["epochs"].append(epoch + 1)
        training_stats["train_losses"].append(train_loss)
        training_stats["eval_losses"].append(eval_loss)
        training_stats["eval_wers"].append(eval_wer)

        # Save best model
        if eval_wer < best_wer:
            best_wer = eval_wer
            training_stats["best_wer"] = best_wer
            training_stats["best_epoch"] = epoch + 1
            save_model(model, output_dir, logger)
            logger.info(f"New best WER: {best_wer:.4f} (saved model)")

    logger.info(f"\n{'='*50}")
    logger.info("Training completed!")
    logger.info(
        f"Best WER: {training_stats['best_wer']:.4f} (Epoch {training_stats['best_epoch']})"
    )
    logger.info(f"{'='*50}")

    return training_stats
