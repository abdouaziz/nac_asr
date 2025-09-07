import logging

from nacasr import setup_logging
import torch 
from nacasr import TextTransform , get_dataloader
from nacasr import NACASR , compute_input_lengths
from nacasr import prepare_ctc_targets
import torch.nn.functional as F 
from nacasr import train_full_pipeline
import torch.nn as nn
from torch.nn import CTCLoss



def main():
    """Main training function"""
    # Setup logging
    logger = setup_logging("nacasr_training.log", logging.INFO)
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize text transform
        text_transform = TextTransform("abdouaziiz/new_benchmark_wolof")
        vocab_size = text_transform.vocab_size()
        logger.info(f"Vocabulary size: {vocab_size}")
        
        # Load datasets
        logger.info("Loading datasets...")
        train_dataloader = get_dataloader(
            "abdouaziiz/new_benchmark_wolof", 
            batch_size=8, 
            num_workers=4, 
         
        )
        eval_dataloader = get_dataloader(
            "abdouaziiz/new_benchmark_wolof", 
            batch_size=8, 
            num_workers=4, 
 
        )
        
        logger.info(f"Training batches: {len(train_dataloader)}")
        logger.info(f"Evaluation batches: {len(eval_dataloader)}")
        
        # Initialize model
        model = NACASR(in_dim=256, vocab_size=vocab_size)
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test model with one batch
        logger.info("Testing model with sample batch...")
        model.to(device)
        model.eval()
        
        sample_batch = next(iter(train_dataloader))
        input_values = sample_batch["input_values"].to(device)
        labels = sample_batch["labels"].to(device)
        
        logger.info(f"Sample input shape: {input_values.shape}")
        logger.info(f"Sample labels shape: {labels.shape}")
        
        with torch.no_grad():
            input_lengths = compute_input_lengths(input_values)
            
            if len(input_values.shape) == 3 and input_values.shape[-1] != 256:
                input_values = input_values.transpose(1, 2)
            
            outputs = model(input_values)
            logger.info(f"Sample output shape: {outputs.shape}")
            
            # Test loss computation
            targets, target_lengths = prepare_ctc_targets(labels, device)
            log_probs = F.log_softmax(outputs, dim=-1).transpose(0, 1)
            
           
            
            loss_fn = CTCLoss(blank=0, zero_infinity=True)
            loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
            logger.info(f"Sample loss: {loss.item():.4f}")
        
        logger.info("Model test completed successfully!")
        
        # Start full training
        training_stats = train_full_pipeline(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            device=device,
            text_transform=text_transform,
            epochs=10,
            lr=1e-3,
            output_dir="./nacasr_model",
            logger=logger
        )
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()