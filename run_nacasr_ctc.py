import logging
from dataclasses import dataclass, field
import torch
from nacasr import (
    setup_logging,
    TextTransform,
    get_dataloader,
    NACASR,
    train_full_pipeline,
)


@dataclass
class TrainingArguments:
    """Configuration class for training arguments"""

    path_or_name: str = field(
        default="abdouaziiz/new_benchmark_wolof",
        metadata={
            "help": "The name of the dataset from Huggingface or the path to load the dataset"
        },
    )

    batch_size: int = field(
        default=8,
        metadata={"help": "The batch size for training the model"},
    )

    num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers for data loading"},
    )

    n_epochs: int = field(
        default=10,
        metadata={"help": "Number of training epochs"},
    )

    lr: float = field(
        default=1e-3,
        metadata={"help": "Learning rate for training"},
    )

    output_dir: str = field(
        default="./nacasr_model",
        metadata={"help": "Directory to save the trained model"},
    )

    log_file: str = field(
        default="nacasr_training.log",
        metadata={"help": "Path to the log file"},
    )

    in_dim: int = field(
        default=256,
        metadata={"help": "Input dimension for the model"},
    )


def validate_training_args(args: TrainingArguments) -> None:
    """Validate training arguments"""
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if args.n_epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    if args.lr <= 0:
        raise ValueError("Learning rate must be positive")
    if args.num_workers < 0:
        raise ValueError("Number of workers cannot be negative")


def main():

    args = TrainingArguments()

    validate_training_args(args)

    logger = setup_logging(args.log_file, logging.INFO)
    logger.info("Starting NACASR training pipeline")
    logger.info(f"Training arguments: {args}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        logger.info("Initializing text transform...")
        text_transform = TextTransform(args.path_or_name)
        vocab_size = text_transform.vocab_size()
        logger.info(f"Vocabulary size: {vocab_size}")

        logger.info("Loading datasets...")
        train_dataloader = get_dataloader(
            dataset_path=args.path_or_name,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        eval_dataloader = get_dataloader(
            dataset_path=args.path_or_name,
            split="validation",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        logger.info(f"Training batches: {len(train_dataloader)}")
        logger.info(f"Evaluation batches: {len(eval_dataloader)}")

        logger.info("Initializing model...")
        model = NACASR(in_dim=args.in_dim, vocab_size=vocab_size)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model initialized with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        model.to(device)
        model.eval()

        try:
            sample_batch = next(iter(train_dataloader))
            input_values = sample_batch["input_values"].to(device)
            labels = sample_batch["labels"].to(device)

            logger.info(f"Sample input shape: {input_values.shape}")
            logger.info(f"Sample labels shape: {labels.shape}")

            with torch.no_grad():
                outputs = model(input_values)
                logger.info(f"Model output shape: {outputs.shape}")

        except Exception as e:
            logger.error(f"Error during model testing: {str(e)}")
            raise

        model.train()

        # Start full training
        logger.info("Starting training pipeline...")
        training_stats = train_full_pipeline(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            device=device,
            text_transform=text_transform,
            epochs=args.n_epochs,
            lr=args.lr,
            output_dir=args.output_dir,
            logger=logger,
        )

        logger.info("Training pipeline completed successfully!")
        logger.info(f"Training statistics: {training_stats}")

        return training_stats

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
