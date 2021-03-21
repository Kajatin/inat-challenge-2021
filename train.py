import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter

import yaml

from src.data.inat import InatDataset
from src.network.inet import iNet
from src.network.functions import train, valid
from src.network.weights import save_model, load_model


# Load configuration file
with open("config.yaml", "r") as f:
    config = yaml.load(f, yaml.FullLoader)

# Set up summary writer
now_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
os.makedirs(config["summary_writer"]["log_dir"], exist_ok=True)
writer = SummaryWriter(
    log_dir=os.path.join(config["summary_writer"]["log_dir"], now_str),
    flush_secs=30
)

# CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create network
net = iNet(config["network"])
net.to(device)
writer.add_graph(net, torch.rand(1, 3, 256, 256))

# Create loss function and optimizer
criterion = nn.CrossEntropyLoss()
criterion.to(device)
optimizer = optim.SGD(
    net.parameters(),
    lr=config["train"]["lr"],
    momentum=config["train"]["momentum"]
)

# Load pretrained model
start_epoch = load_model(net, optimizer, config["train"]["pretrained_path"])

# Create datasets
train_dataset = InatDataset(
    config["datasets"]["train"],
    transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomAffine(20, (0.2, 0.2), (0.8, 1.2), (-10, 10, -10, 10)),
        transforms.RandomGrayscale(0.05),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.1),
        transforms.RandomErasing(0.1),
        transforms.Resize(300),
        transforms.CenterCrop(config["network"]["input_dims"]),
        transforms.Normalize(
            config["datasets"]["normalize"]["mean"],
            config["datasets"]["normalize"]["std"]
        )
    ])
)
valid_dataset = InatDataset(
    config["datasets"]["valid"],
    transforms.Compose([
        transforms.Resize(config["network"]["input_dims"]),
        transforms.ToTensor(),
        transforms.Normalize(
            config["datasets"]["normalize"]["mean"],
            config["datasets"]["normalize"]["std"]
        )
    ])
)

# Create dataloaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config["dataloaders"]["train"]["batch_size"],
    shuffle=config["dataloaders"]["train"]["shuffle"]
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=config["dataloaders"]["valid"]["batch_size"],
    shuffle=config["dataloaders"]["valid"]["shuffle"]
)

# Create train/validation loops
for e in range(start_epoch, config["train"]["end_epoch"]):
    train(
        net,
        train_loader,
        optimizer,
        criterion,
        device,
        e,
        writer
    )

    save_model(net, optimizer, e)

    valid(
        net,
        valid_loader,
        criterion,
        device,
        e,
        writer
    )

writer.close()
