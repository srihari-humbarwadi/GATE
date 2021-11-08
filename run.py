import argparse
import glob
import logging
import os
import tarfile

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import WandbLogger
from rich import print
from rich.logging import RichHandler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from datasets.dataset_loading_hub import load_dataset
from models import model_zoo
from models.system_models import contrastive_logits_labels
from utils.arg_parsing import add_extra_option_args, process_args
from utils.storage import (build_experiment_folder, restore_model,
                           save_checkpoint)

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logging = logging.getLogger("rich")


def compute_accuracy(logits, targets):
    acc = targets == logits.argmax(-1)
    return torch.mean(acc.type(torch.float32)) * 100


def get_base_argument_parser():
    parser = argparse.ArgumentParser()
    # data and I/O
    parser.add_argument("--dataset_name", type=str, default="tali")
    parser.add_argument("--rescan_dataset_files", default=False, action="store_true")
    parser.add_argument("--offline_mode", default=False, action="store_true")
    parser.add_argument("--upload_model_weights", default=False, action="store_true")
    parser.add_argument("--data_filepath", type=str, default="data/sample_dataset")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--exclude_modalities", nargs="+")
    # 'video, audio, text, image'
    parser.add_argument("--restrict_train_set_size", type=int, default=None)
    parser.add_argument("--num_data_provider_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--image_height", type=int, default=224)
    parser.add_argument("--image_width", type=int, default=224)
    parser.add_argument("--num_video_frames_per_datapoint", type=int, default=30)
    parser.add_argument("--num_audio_frames_per_datapoint", type=int, default=44000)
    parser.add_argument("--text_context_length", type=int, default=77)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    # logging
    parser.add_argument("--experiment_name", type=str, default="dev")
    parser.add_argument("--logs_path", type=str, default="log")
    parser.add_argument("--filepath_to_arguments_json_config", type=str, default=None)

    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="CosineAnnealing",
        help="Scheduler for learning rate annealing: CosineAnnealing | MultiStep",
    )
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=[60, 120, 160],
        help="Multi step scheduler annealing milestones",
    )
    parser.add_argument("--optim", type=str, default="Adam", help="Optimizer?")
    parser.add_argument(
        "--min_learning_rate",
        type=float,
        default=0.0,
        help="Min learning rate the scheduler should anneal towards",
    )

    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser = add_extra_option_args(parser)

    return parser


############################################################################## Training
metrics_to_track = {
    "cross_entropy": lambda x, y: F.cross_entropy(input=x, target=y),
    "accuracy": lambda x, y: compute_accuracy(x, y),
}


class LightningExperiment(pl.LightningModule):
    def __init__(self, dummy_set_loader, args):
        super(LightningExperiment, self).__init__()
        # (**args.model)
        self.save_hyperparameters()

        dummy_batch = next(iter(dummy_set_loader))

        self.model = model_zoo[args.model.type](**args.model)

        self.exclude_modalities = self.hparams.args.exclude_modalities
        self.args = self.hparams.args

        if self.exclude_modalities:
            if "video" in self.exclude_modalities:
                dummy_batch["video"] = None

            if "audio" in self.exclude_modalities:
                dummy_batch["audio"] = None

            if "image" in self.exclude_modalities:
                dummy_batch["image"] = None

            if "text" in self.exclude_modalities:
                dummy_batch["text"] = None

        _, _ = self.model.forward(
            image_input=dummy_batch["image"],
            audio_input=dummy_batch["audio"],
            text_input=dummy_batch["text"],
            video_input=dummy_batch["video"],
        )

    @staticmethod
    def add_model_specific_args(args):
        return args

    def forward(self, x_image, x_audio, x_text, x_video):

        outputs, cosine_similarities = self.model.forward(
            image_input=x_image,
            audio_input=x_audio,
            text_input=x_text,
            video_input=x_video,
        )

        logits, _ = contrastive_logits_labels(
            torch.stack(list(cosine_similarities.values()))
        )

        return logits

    def collect_metrics(self, logits, targets, phase_name):

        for metric_key, metric_function in metrics_to_track.items():
            for measurement_key, measurement_value, target_value in zip(
                logits.keys(), logits.values(), targets
            ):
                self.log(
                    name=f"{phase_name}/{metric_key}_{measurement_key}",
                    value=metric_function(
                        measurement_value.detach(), target_value.detach()
                    ),
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                )

            self.log(
                name=f"{phase_name}/overall_{metric_key}",
                value=metric_function(
                    torch.stack(list(logits.values())).detach(), targets.detach()
                ),
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )

    def training_step(self, batch, batch_idx):

        if self.exclude_modalities:
            if "video" in self.exclude_modalities:
                batch["video"] = None

            if "audio" in self.exclude_modalities:
                batch["audio"] = None

            if "image" in self.exclude_modalities:
                batch["image"] = None

            if "text" in self.exclude_modalities:
                batch["text"] = None

        outputs, cosine_similarities = self.model.forward(
            image_input=batch["image"],
            audio_input=batch["audio"],
            text_input=batch["text"],
            video_input=batch["video"],
        )

        logits, targets = contrastive_logits_labels(
            torch.stack(list(cosine_similarities.values()))
        )
        self.collect_metrics(
            logits=cosine_similarities, targets=targets, phase_name="train"
        )
        return {"loss": F.cross_entropy(input=logits, target=targets)}

    def validation_step(self, batch, batch_idx):
        if self.exclude_modalities:
            if "video" in self.exclude_modalities:
                batch["video"] = None

            if "audio" in self.exclude_modalities:
                batch["audio"] = None

            if "image" in self.exclude_modalities:
                batch["image"] = None

            if "text" in self.exclude_modalities:
                batch["text"] = None

        outputs, cosine_similarities = self.model.forward(
            image_input=batch["image"],
            audio_input=batch["audio"],
            text_input=batch["text"],
            video_input=batch["video"],
        )

        logits, targets = contrastive_logits_labels(
            torch.stack(list(cosine_similarities.values()))
        )

        self.collect_metrics(
            logits=cosine_similarities, targets=targets, phase_name="validation"
        )

    def test_step(self, batch, batch_idx):

        if self.exclude_modalities:
            if "video" in self.exclude_modalities:
                batch["video"] = None

            if "audio" in self.exclude_modalities:
                batch["audio"] = None

            if "image" in self.exclude_modalities:
                batch["image"] = None

            if "text" in self.exclude_modalities:
                batch["text"] = None

        outputs, cosine_similarities = self.model.forward(
            image_input=batch["image"],
            audio_input=batch["audio"],
            text_input=batch["text"],
            video_input=batch["video"],
        )

        logits, targets = contrastive_logits_labels(
            torch.stack(list(cosine_similarities.values()))
        )

        self.collect_metrics(
            logits=cosine_similarities, targets=targets, phase_name="test"
        )

    def configure_optimizers(self):
        args = self.args
        if args.optim.lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )

        if args.scheduler == "CosineAnnealing":
            scheduler = CosineAnnealingLR(
                optimizer=optimizer,
                T_max=args.max_epochs,
                eta_min=args.min_learning_rate,
            )
        else:
            scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2)
        logging.info(
            f"optimizer: {repr(optimizer)},"
            f"max-epochs: {args.max_epochs},"
            f"current-epoch: {self.current_epoch}",
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    argument_parser = get_base_argument_parser()

    argument_parser = pl.Trainer.add_argparse_args(argument_parser)
    argument_parser = LightningExperiment.add_model_specific_args(argument_parser)
    args = process_args(argument_parser)

    # save_dir: Optional[str] = None,
    # offline: Optional[bool] = False,
    # id: Optional[str] = None,
    # anonymous: Optional[bool] = None,
    # version: Optional[str] = None,
    # project: Optional[str] = None,
    # log_model: Optional[bool] = False,
    # experiment = None,
    # prefix: Optional[str] = "",
    # sync_step: Optional[bool] = None,

    ############################################################################# Admin
    saved_models_filepath, logs_filepath, images_filepath = build_experiment_folder(
        experiment_name=args.experiment_name, log_path=args.logs_path
    )

    ############################################################################## Data

    #  if you have an environment variable set, prioritise this over the default
    # argparse option
    environment_data_filepath = os.environ.get("PYTORCH_DATA_LOC")
    if environment_data_filepath is not None and os.path.exists(
        environment_data_filepath
    ):
        logging.warning(
            f"You have a data filepath set in your environment: "
            f"{environment_data_filepath}. This will override "
            f"argparse. "
        )
        data_filepath = environment_data_filepath
    else:
        data_filepath = args.data_filepath
    (
        dummy_set_loader,
        train_set_loader,
        val_set_loader,
        test_set_loader,
        train_set,
        val_set,
        test_set,
        data_shape,
    ) = load_dataset(
        dataset=args.dataset_name,
        data_filepath=args.data_filepath,
        image_height=args.image_height,
        image_width=args.image_width,
        exclude_modalities=args.exclude_modalities,
        batch_size=args.batch_size,
        test_batch_size=args.eval_batch_size,
        num_workers=args.num_data_provider_workers,
        download=False,
        num_video_frames_per_datapoint=args.num_video_frames_per_datapoint,
        num_audio_frames_per_datapoint=args.num_audio_frames_per_datapoint,
        text_context_length=args.text_context_length,
        prefetch_factor=args.prefetch_factor,
        rescan_dataset_files=args.rescan_dataset_files,
        restrict_train_set_size=args.restrict_train_set_size,
    )

    seed_everything(args.seed, workers=True)

    # Always save a snapshot of the current state of the code. I've found this helps
    # immensely if you find that one of your many experiments was actually quite good
    # but you forgot what you did

    snapshot_filename = f"{saved_models_filepath}/snapshot.tar.gz"
    filetypes_to_include = [".py"]
    all_files = []
    for filetype in filetypes_to_include:
        all_files += glob.glob("**/*.py", recursive=True)
    with tarfile.open(snapshot_filename, "w:gz") as tar:
        for file in all_files:
            tar.add(file)

    wandb_instance = WandbLogger(
        project="TALI",
        save_dir=f"{args.logs_path}/{args.experiment_name}/",
        offline=args.offline_mode,
        log_model=args.upload_model_weights,
    )  # your credentials

    trainer_args = pl.Trainer.parse_argparser(arg_parser=argument_parser)

    checkpoint_loss_callback = ModelCheckpoint(
        monitor="validation/overall_cross_entropy_epoch",
        dirpath=f"{args.logs_path}/{args.experiment_name}/checkpoints/",
        filename="model-loss-{epoch:02d}",
        save_on_train_epoch_end=True,
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        every_n_val_epochs=1,
        mode="min",
    )

    checkpoint_acc_callback = ModelCheckpoint(
        monitor="validation/overall_accuracy_epoch",
        dirpath=f"{args.logs_path}/{args.experiment_name}/checkpoints/",
        filename="model-accuracy-{epoch:02d}",
        save_on_train_epoch_end=True,
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        every_n_val_epochs=1,
        mode="max",
    )

    experiment = LightningExperiment(dummy_set_loader=dummy_set_loader, args=args)

    logging.info(
        f"max-epochs: {args.max_epochs}," f"current-epoch: {experiment.current_epoch}",
    )

    summary_callback = ModelSummary(model=experiment, max_depth=1)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    logging.info(summary_callback)

    if args.resume_from_checkpoint == "scratch":
        args.resume_from_checkpoint = (
            f"{args.logs_path}/{args.experiment_name}" f"/checkpoints/ "
        )

    trainer = pl.Trainer.from_argparse_args(
        args=trainer_args,
        logger=wandb_instance,
        log_every_n_steps=1,
        default_root_dir=f"{args.logs_path}/{args.experiment_name}/",
        resume_from_checkpoint=args.resume_from_checkpoint,
        callbacks=[checkpoint_loss_callback, checkpoint_acc_callback, lr_monitor],
    )

    trainer.fit(experiment, train_set_loader, val_set_loader)

    test_results = trainer.test(dataloaders=test_set_loader)
