import os
import torch
import logging
from typing import Tuple , List
from nacasr.dataset import TextTransform



def setup_logging(log_file: str = "training.log", log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        log_level: Logging level
    
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_path}")
    
    return logger


def prepare_ctc_targets(labels: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare targets for CTC loss by removing padding tokens and flattening
    
    Args:
        labels: Label tensor with padding tokens (-100)
        device: Device to place tensors on
    
    Returns:
        Tuple of (flattened_targets, target_lengths)
    """
    targets = []
    target_lengths = []
    
    for i in range(labels.shape[0]):
        # Remove padding tokens (-100)
        target = labels[i][labels[i] != -100]
        targets.append(target)
        target_lengths.append(len(target))
    
    # Flatten targets and create lengths tensor
    if targets:
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
    else:
        targets = torch.empty(0, dtype=torch.long, device=device)
        target_lengths = torch.empty(0, dtype=torch.long, device=device)
    
    return targets, target_lengths


def greedy_decoder(outputs: torch.Tensor, labels: torch.Tensor, text_transform: TextTransform, 
                  blank_label: int = 0, collapse_repeated: bool = True) -> Tuple[List[str], List[str]]:
    """
    Greedy CTC decoder
    
    Args:
        outputs: Model output logits (B, T, vocab_size)
        labels: Ground truth labels (B, max_label_length)
        text_transform: Text transformation object
        blank_label: Blank token id for CTC
        collapse_repeated: Whether to collapse repeated predictions
    
    Returns:
        Tuple of (decoded_predictions, target_strings)
    """
    arg_maxes = torch.argmax(outputs, dim=2)
    decodes = []
    targets = []
    
    for i, args in enumerate(arg_maxes):
        # Decode predictions
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        
        # Convert to text
        decodes.append(text_transform.int_to_text(decode))
        
        # Get target text (remove padding)
        target = labels[i][labels[i] != -100].tolist()
        targets.append(text_transform.int_to_text(target))
    
    return decodes, targets


def save_model(model: torch.nn.Module, output_dir: str, logger: logging.Logger) -> None:
    """
    Save model state dict to output directory
    
    Args:
        model: Model to save
        output_dir: Output directory path
        logger: Logger instance
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        model_path = os.path.join(output_dir, "model.pth")
        torch.save(model_to_save.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")

