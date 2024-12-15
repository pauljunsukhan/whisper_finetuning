# File: main.py

"""Main entry point for the Whisper Fine-Tuning Experiment."""

import os
import yaml
import torch
import torch.distributed as dist
import argparse
import sys
from pathlib import Path

# Use absolute imports since we're running as a module
from experiments.experiment_002.components import initialize_components
from experiments.experiment_002.components.logger import LoggerManager
from experiments.experiment_002.components.base import ExperimentError

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Loaded configuration as a dictionary.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_distributed() -> tuple[int, int]:
    """Set up distributed training.

    Detects if the environment variables for distributed training are set and initializes the process group.

    Returns:
        tuple: A tuple containing (rank, world_size).
    """
    if torch.cuda.is_available():
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            dist.init_process_group('nccl')
        else:
            rank = 0
            world_size = 1
    else:
        rank = 0
        world_size = 1

    return rank, world_size

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # Get the directory where the script is located
    script_dir = Path(__file__).resolve().parent
    default_config = script_dir / 'config.yaml'

    parser = argparse.ArgumentParser(description="Fine-Tune Whisper Large V3 Model")
    parser.add_argument(
        '--config',
        type=str,
        default=str(default_config),
        help='Path to the configuration YAML file.'
    )
    return parser.parse_args()

def main():
    """Main function to execute the Whisper fine-tuning experiment."""
    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Configuration file not found at {config_path}")
        sys.exit(1)
    config = load_config(str(config_path))

    # Configure PyTorch threading based on config
    threading_config = config.get('training', {}).get('threading', {})
    num_threads = threading_config.get('num_threads', 14)
    num_interop_threads = threading_config.get('num_interop_threads', 1)

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)

    # Setup distributed training
    rank, world_size = setup_distributed()

    # Initialize temporary logger for early stages
    temp_logger = LoggerManager.load_temp_logger()
    temp_logger.log_info(f"Starting experiment with rank {rank}/{world_size}")

    try:
        # Augment configuration with distributed training details
        config['distributed'] = {'rank': rank, 'world_size': world_size}

        # Initialize all components
        state_manager, logger_manager, model_manager, data_manager, trainer = initialize_components(config)

        # Log threading configuration using the main logger
        logger_manager.log_info(f"PyTorch set_num_threads: {torch.get_num_threads()}")
        logger_manager.log_info(f"PyTorch set_num_interop_threads: {torch.get_num_interop_threads()}")

        # Log state information
        state_manager.log_state()

        # Prepare datasets
        data_manager.prepare_datasets(
            test_size=config['dataset']['split']['test_size'],
            seed=config['dataset']['split']['seed']
        )

        # Start training
        trainer.train()

        # Final evaluation
        if state_manager.is_main_process():
            metrics = trainer.evaluate()
            logger_manager.log_info(f"Final evaluation metrics: {metrics}")

    except ExperimentError as ee:
        temp_logger.log_error(f"Experiment failed due to configuration error: {str(ee)}")
        sys.exit(1)
    except Exception as e:
        temp_logger.log_error(f"Experiment failed: {str(e)}")
        raise
    finally:
        # Clean up distributed training resources
        if torch.cuda.is_available() and world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()
