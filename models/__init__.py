import os
import torch
from models.resnet_c2d import BaseModel
from models.transformer import TransformerModel


def build_model(cfg):
    if cfg.MODEL.EMBEDDER_TYPE == "transformer":
        model = TransformerModel(cfg)
    else:
        model = BaseModel(cfg)
    return model


def save_checkpoint(cfg, model, optimizer, epoch):
    path = os.path.join(cfg.LOGDIR, "checkpoints")
    os.makedirs(path, exist_ok=True)
    ckpt_path = os.path.join(path, f"checkpoint_epoch_{epoch:05d}.pth")
    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg,
    }
    torch.save(checkpoint, ckpt_path)


def load_checkpoint(cfg, model, optimizer):
    path = os.path.join(cfg.LOGDIR, "checkpoints")
    if os.path.exists(path):
        names = [f for f in os.listdir(path) if "checkpoint" in f]
        if len(names) > 0:
            name = sorted(names)[-1]
            ckpt_path = os.path.join(path, name)
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint["model_state"], False)
            # optimizer.load_state_dict(checkpoint["optimizer_state"])
            # cfg.update(checkpoint["cfg"])
            return checkpoint["epoch"]
    return 0
