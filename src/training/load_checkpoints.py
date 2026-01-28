import torch
import torch.nn as nn
from pathlib import Path
from configs.training_params import TrainingParams


def load_model_checkpoint(model: nn.Module, checkpoints_file_path: str) -> nn.Module:
    """
    Loads model weights from checkpoint file.
    :param model: Instance of predictive model.
    :param checkpoints_file_path: Name of checkpoints file.
    :return: Loaded model with weights from training.
    """
    full_path: str = f"{TrainingParams.DEFAULT_CHECKPOINTS_DIR}/{checkpoints_file_path}"
    weights_path = Path(full_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    else:
        loaded_weights = torch.load(f=weights_path, weights_only=True)
        model.load_state_dict(loaded_weights)

    return model
