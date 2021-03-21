import os
from datetime import datetime

import torch


def save_model(model, optimizer, epoch, path=None):
    if not path:
        path = "models/model_{}_epoch_{}".format(
            datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
            epoch
        )
    if not path.endswith(".pth"):
        path = path + ".pth"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        },
        path
    )


def load_model(model, optimizer, path):
    if not os.path.isfile(path):
        return 0

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"] + 1
