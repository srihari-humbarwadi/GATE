import argparse
import logging

import torch
import torch.nn.functional as F
from gate.architectures import AutoConv2DTransformersFlatten
from datasets import CIFAR10Loader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

parser = argparse.ArgumentParser()
# data and I/O
parser.add_argument("--dataset_name", type=str, default="tali")
parser.add_argument("--rescan_dataset_files", default=False, action="store_true")
parser.add_argument("--data_filepath", type=str, default="data/sample_dataset")

parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--exclude_modalities", nargs="+")
# 'video, audio, text, image'
parser.add_argument("--num_data_provider_workers", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--image_height", type=int, default=224)
parser.add_argument("--image_width", type=int, default=224)
parser.add_argument("--num_video_frames_per_datapoint", type=int, default=30)
parser.add_argument("--num_audio_frames_per_datapoint", type=int, default=44000)
parser.add_argument("--text_context_length", type=int, default=77)
parser.add_argument("--prefetch_factor", type=int, default=2)
args = parser.parse_args()

data = CIFAR10Loader()
train_data, val_data, test_data, num_labels = data.get_data(
    data_filepath="../../datasets",
    val_set_percentage=0.1,
    random_split_seed=1,
    download=True,
)
train_data_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=1024,
    shuffle=True,
    num_workers=8,
    pin_memory=False,
    prefetch_factor=2,
    collate_fn=None,
    persistent_workers=False,
)

image_model = AutoConv2DTransformersFlatten(
    num_classes=num_labels,
    grid_patch_size=4,
    transformer_num_filters=128,
    transformer_num_layers=4,
    transformer_num_heads=4,
    transformer_dim_feedforward=512,
    stem_conv_bias=False,
)

x, y = next(iter(train_data_loader))
out_image_output, _ = image_model.forward(x=x)
logging.info(f"{out_image_output.shape}, {x.shape}")


def test_training_multi_modal_model(data_loader, system, num_epochs):
    system = system.to(torch.cuda.current_device())
    system.train()
    optim = Adam(params=system.parameters(), lr=0.001, weight_decay=0.00001)
    lr_scheduler = CosineAnnealingLR(
        optimizer=optim,
        T_max=num_epochs,
        eta_min=0.0000001,
    )
    for epoch in range(num_epochs):
        for idx, (x, y) in enumerate(data_loader):
            x = x.to(torch.cuda.current_device())
            y = y.to(torch.cuda.current_device())
            logits, _ = image_model.forward(x)
            loss = F.cross_entropy(logits, y.to(torch.cuda.current_device()))
            optim.zero_grad()

            system.zero_grad()
            loss.backward()
            optim.step()

            logging.info(
                f"{epoch}:::{idx}: loss: {loss}, " f"lr: {lr_scheduler.get_last_lr()}"
            )
        lr_scheduler.step()


test_training_multi_modal_model(
    data_loader=train_data_loader, system=image_model, num_epochs=300
)
