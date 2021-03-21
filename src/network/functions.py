import torch
from tqdm import tqdm

from src.utils.average_meter import AverageMeter
from src.network.topk import topk


def train(
    net,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch,
    writer
):
    losses = AverageMeter()
    top1error = AverageMeter()
    top5error = AverageMeter()

    net.train()

    for i, data in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        unit="batch",
        postfix={
            "epoch": epoch
        }
    ):
        images = data["image"].to(device)
        labels = data["label"].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        top1error.update(topk(outputs.detach(), labels.detach(), 1), labels.shape[0])
        top5error.update(topk(outputs.detach(), labels.detach(), 5), labels.shape[0])

        writer.add_scalar("train/loss/local", losses.val, (epoch*len(dataloader)+i))
        writer.add_scalar("train/top1error/local", top1error.val, (epoch*len(dataloader)+i))
        writer.add_scalar("train/top5error/local", top5error.val, (epoch*len(dataloader)+i))

    writer.add_scalar("train/loss", losses.avg, epoch)
    writer.add_scalar("train/top1error", top1error.avg, epoch)
    writer.add_scalar("train/top5error", top5error.avg, epoch)


@torch.no_grad()
def valid(
    net,
    dataloader,
    criterion,
    device,
    epoch,
    writer
):
    losses = AverageMeter()
    top1error = AverageMeter()
    top5error = AverageMeter()

    net.eval()

    for i, data in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        unit="batch",
        postfix={
            "epoch": epoch
        }
    ):
        images = data["image"].to(device)
        labels = data["label"].to(device)

        # Forward
        outputs = net(images)
        loss = criterion(outputs, labels)

        losses.update(loss.item())
        top1error.update(topk(outputs, labels, 1), labels.shape[0])
        top5error.update(topk(outputs, labels, 5), labels.shape[0])

        writer.add_scalar("valid/loss/local", losses.val, (epoch*len(dataloader)+i))
        writer.add_scalar("valid/top1error/local", top1error.val, (epoch*len(dataloader)+i))
        writer.add_scalar("valid/top5error/local", top5error.val, (epoch*len(dataloader)+i))

    writer.add_scalar("valid/loss", losses.avg, epoch)
    writer.add_scalar("valid/top1error", top1error.avg, epoch)
    writer.add_scalar("valid/top5error", top5error.avg, epoch)
