import argparse
import os
from typing import Any, Dict, List

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from allennlp.data import Vocabulary

from updown.config import Config
from updown.data.datasets import (
    TrainingDataset,
    EvaluationDataset,
    EvaluationDatasetWithConstraints,
)
from updown.data.meta_batch_sampler import MetaBatchSampler
from updown.models import UpDownCaptioner
from updown.types import Prediction
from updown.utils.checkpointing import CheckpointManager
from updown.utils.common import cycle
from updown.utils.evalai import NocapsEvaluator
from updown.utils.constraints import add_constraint_words_to_vocabulary
from updown.modules.meta import Meta


parser = argparse.ArgumentParser("Train an UpDown Captioner on COCO train2017 split.")
parser.add_argument(
    "--config", required=True, help="Path to a config file with all configuration parameters."
)
parser.add_argument(
    "--config-override",
    default=[],
    nargs="*",
    help="A sequence of key-value pairs specifying certain config arguments (with dict-like "
    "nesting) using a dot operator. The actual config will be updated and recorded in "
    "the serialization directory.",
)

parser.add_argument_group("Compute resource management arguments.")
parser.add_argument(
    "--gpu-ids", required=True, nargs="+", type=int, help="List of GPU IDs to use (-1 for CPU)."
)
parser.add_argument(
    "--cpu-workers", type=int, default=0, help="Number of CPU workers to use for data loading."
)
parser.add_argument(
    "--in-memory", action="store_true", help="Whether to load image features in memory."
)

parser.add_argument_group("Checkpointing related arguments.")
parser.add_argument(
    "--skip-validation",
    action="store_true",
    help="Whether to skip validation and simply serialize checkpoints. This won't track the "
    "best performing checkpoint (obviously). useful for cases where GPU server does not have "
    "internet access and/or checkpoints are validation externally.",
)
parser.add_argument(
    "--serialization-dir",
    default="checkpoints/experiment/meta",
    help="Path to a (non-existent) directory for serializing checkpoints and tensorboard logs.",
)
parser.add_argument(
    "--checkpoint-every",
    default=1000,
    type=int,
    help="Save a checkpoint after every this many epochs/iterations.",
)
parser.add_argument(
    "--start-from-checkpoint",
    default="",
    help="Path to load checkpoint and continue training [only supported for module_training].",
)


if __name__ == "__main__":
    # --------------------------------------------------------------------------------------------
    #   INPUT ARGUMENTS AND CONFIG
    # --------------------------------------------------------------------------------------------
    _A = parser.parse_args()

    # Create a config with default values, then override from config file, and _A.
    # This config object is immutable, nothing can be changed in this anymore.
    _C = Config(_A.config, _A.config_override)

    # Print configs and args.
    print(_C)
    for arg in vars(_A):
        print("{:<20}: {}".format(arg, getattr(_A, arg)))

    # Create serialization directory and save config in it.
    os.makedirs(_A.serialization_dir, exist_ok=True)
    _C.dump(os.path.join(_A.serialization_dir, "config.yml"))

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    # These five lines control all the major sources of randomness.
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)
    torch.cuda.manual_seed_all(_C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set device according to specified GPU ids.
    device = torch.device("cuda:0")

    # --------------------------------------------------------------------------------------------
    #   INSTANTIATE VOCABULARY, DATALOADER, MODEL, OPTIMIZER
    # --------------------------------------------------------------------------------------------

    vocabulary = Vocabulary.from_files(_C.DATA.VOCABULARY)

    # If we wish to use CBS during evaluation or inference, expand the vocabulary and add
    # constraint words derived from Open Images classes.
    if _C.MODEL.USE_CBS:
        vocabulary = add_constraint_words_to_vocabulary(
            vocabulary, wordforms_tsvpath=_C.DATA.CBS.WORDFORMS
        )

    train_dataset = TrainingDataset.from_config(_C, vocabulary=vocabulary, in_memory=_A.in_memory)
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=_A.cpu_workers,
        collate_fn=train_dataset.collate_fn,
        sampler=MetaBatchSampler.from_config(_C,vocabulary=vocabulary, in_memory=False)
    )
    # Make dataloader cyclic for sampling batches perpetually.
    train_dataloader = cycle(train_dataloader, device)

    # TODO setup args
    maml = Meta(_C, vocabulary).to(device)

    # --------------------------------------------------------------------------------------------
    #  BEFORE TRAINING STARTS
    # --------------------------------------------------------------------------------------------

    # Tensorboard summary writer for logging losses and metrics.
    # tensorboard_writer = SummaryWriter(logdir=_A.serialization_dir)

    # --------------------------------------------------------------------------------------------
    #   TRAINING LOOP
    # --------------------------------------------------------------------------------------------
    for iteration in tqdm(range(start_iteration, _C.OPTIM.NUM_ITERATIONS + 1)):

        # keys: {"image_id", "image_features", "caption_tokens"}
        x, y = next(train_dataloader)
        x, y = x.to(device), y.to(device)

        loss = maml(x,y)
        if (iteration%30) == 0:
            print(loss)

        if (iteration%500) == 0:
            losses_all_test = []
            for i in range(100):
                x, y = next(train_dataloader)
                x, y = x.to(device), y.to(device)
                loss = maml.finetunning(x, y)
                losses_all_test.append(loss)
            losses = np.array(losses_all_test).mean(axis=0).astype(np.float16)
            print("Test acc:", losses)
            # Log loss and learning rate to tensorboard.
            #tensorboard_writer.add_scalar("loss", batch_loss, iteration)

        # ----------------------------------------------------------------------------------------
        #   VALIDATION
        # ----------------------------------------------------------------------------------------
        if iteration % _A.checkpoint_every == 0:
            pass