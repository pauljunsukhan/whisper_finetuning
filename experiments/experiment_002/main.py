"""Main entry point for experiment"""

import os
import yaml
import torch
import torch.distributed as dist
from pathlib import Path

from .components import initialize_components
from .components.logger import LoggerManager

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_distributed() -> tuple[int, int]:
    """Setup distributed training

    Returns:
        Tuple of (rank, world_size)
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

def main():
    """Main entry point"""
    # Load configuration
    config_path = Path(__file__).parent / 'config.yaml'
    config = load_config(str(config_path))

    # **Configure PyTorch threading based on config**
    threading_config = config.get('training', {}).get('threading', {})
    num_threads = threading_config.get('num_threads', 14)
    num_interop_threads = threading_config.get('num_interop_threads', 1)

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)

    # **Log the threading configuration**
    print(f"PyTorch set_num_threads: {torch.get_num_threads()}")
    print(f"PyTorch set_num_interop_threads: {torch.get_num_interop_threads()}")

    # Setup distributed training
    rank, world_size = setup_distributed()

    # Initialize temporary logger for early stages
    temp_logger = LoggerManager.load_temp_logger()
    temp_logger.log_info(f"Starting experiment with rank {rank}/{world_size}")

    try:
        # Add rank to config for component initialization
        config['distributed'] = {'rank': rank, 'world_size': world_size}

        # Initialize components
        state_manager, logger_manager, model_manager, data_manager, trainer = initialize_components(config)

        # Prepare datasets
        data_manager.prepare_datasets(
            test_size=config['dataset']['split']['test_size'],
            seed=config['dataset']['split']['seed']
        )

        # **No need to create DataLoaders manually**
        # The TrainerManager handles DataLoader creation internally using the provided datasets

        # Start training
        trainer.train()

        # Final evaluation
        if state_manager.is_main_process():
            metrics = trainer.evaluate()
            logger_manager.log_info(f"Final evaluation metrics: {metrics}")

    except Exception as e:
        temp_logger.log_error(f"Experiment failed: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available() and world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()
