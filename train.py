# coding=utf-8

import os
import pprint
import torch
import random
from tqdm import tqdm, trange
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import utils.distributed as du
import utils.logging as logging
from utils.parser import parse_args, load_config, setup_train_dir
from models import build_model, save_checkpoint, load_checkpoint
from utils.optimizer import construct_optimizer, construct_scheduler, get_lr
from Meview import Meview
from torch.utils.data import DataLoader
from algos import get_algo
from evaluation import get_tasks
from sklearn.metrics import f1_score, recall_score, confusion_matrix


logger = logging.get_logger(__name__)


def classify_evaluation(gt, pred):
    gt = gt.cpu()
    pred = pred.cpu()
    f1 = f1_score(gt, pred, zero_division=1)
    recall = recall_score(gt, pred, zero_division=1)
    return f1, recall


def train(cfg, train_loader, model, optimizer, scheduler, algo, cur_epoch, summary_writer):
    model.train()
    optimizer.zero_grad()
    data_size = len(train_loader)
    total_loss = {}

    if du.is_root_proc():
        train_loader = tqdm(train_loader,
                            total=len(train_loader),
                            desc=f'[train-000] loss=0.000, f1=0.000, recall=0.000')

    groundTruth, predicts = torch.Tensor(), torch.Tensor()
    for cur_iter, (videos, _labels, video_masks) in enumerate(train_loader):
        optimizer.zero_grad()
        loss_dict, gts, preds = algo.compute_loss(model, videos, _labels, video_masks)
        loss = loss_dict["loss"]
        # Perform the backward pass.
        loss.backward()

        groundTruth = torch.cat((groundTruth.cuda(), gts.cuda()))
        predicts = torch.cat((predicts.cuda(), preds.cuda()))
        f1, recall = classify_evaluation(groundTruth, predicts)
        train_loader.set_description(f"[train-{cur_iter:03d}] {loss=:.3f}, {f1=:.3f}, {recall=:.3f}")
        train_loader.refresh()

        # Update the parameters.
        if cfg.OPTIMIZER.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.OPTIMIZER.GRAD_CLIP)
        optimizer.step()

        for key in loss_dict:
            loss_dict[key][torch.isnan(loss_dict[key])] = 0
            if key not in total_loss:
                total_loss[key] = 0
            # total_loss[key] += du.all_reduce([loss_dict[key]]
            #                                  )[0].item() / data_size
    summary_writer.add_scalar('train/learning_rate',
                              get_lr(optimizer)[0], cur_epoch)
    for key in total_loss:
        summary_writer.add_scalar(f'train/{key}', total_loss[key], cur_epoch)

    f1, recall = classify_evaluation(groundTruth.cuda(), predicts.cuda())
    tn, fp, fn, tp = confusion_matrix(groundTruth.cpu(), predicts.cpu(), labels=[0,1]).ravel()
    print(f"[train results] {tp=:2d}, {tn=:2d}, {fp=:2d}, {fn=:2d}, {f1=:.3f}, {recall=:.3f}")


    logger.info("epoch {}, train loss: {:.3f}".format(
        cur_epoch, total_loss["loss"]))

    if cur_epoch != cfg.TRAIN.MAX_EPOCHS-1:
        scheduler.step()


def val(cfg, val_loader, model, algo, cur_epoch, summary_writer):
    model.eval()
    data_size = len(val_loader)
    total_loss = {}

    groundTruth, predicts = torch.Tensor(), torch.Tensor()
    with torch.no_grad():
        for cur_iter, (videos, labels, video_masks) in enumerate(val_loader):
            loss_dict, gts, preds = algo.compute_loss(
                model, videos, labels, video_masks, training=False)

            groundTruth = torch.cat((groundTruth.cuda(), gts.cuda()))
            predicts = torch.cat((predicts.cuda(), preds.cuda()))
            for key in loss_dict:
                loss_dict[key][torch.isnan(loss_dict[key])] = 0
                if key not in total_loss:
                    total_loss[key] = 0
                # total_loss[key] += du.all_reduce([loss_dict[key]]
                #                                  )[0].item() / data_size

    f1, recall = classify_evaluation(groundTruth.cuda(), predicts.cuda())
    tn, fp, fn, tp = confusion_matrix(groundTruth.cpu(), predicts.cpu(), labels=[0,1]).ravel()
    print(f"[-val- results] {tp=:2d}, {tn=:2d}, {fp=:2d}, {fn=:2d}, {f1=:.3f}, {recall=:.3f}\n")

    for key in total_loss:
        summary_writer.add_scalar(f'val/{key}', total_loss[key], cur_epoch)
    logger.info("epoch {}, train loss: {:.3f}".format(
        cur_epoch, total_loss["loss"]))


def main():
    args = parse_args()
    cfg = load_config(args)
    setup_train_dir(cfg.LOGDIR)
    cfg.NUM_GPUS = torch.cuda.device_count()  # num_gpus_per_machine
    args.world_size = int(os.getenv('WORLD_SIZE', 0))  # total_gpus
    if os.environ.get('OMPI_COMM_WORLD_SIZE') is None:
        args.rank = args.local_rank
    else:
        args.node_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
        args.rank = args.node_rank * torch.cuda.device_count() + args.local_rank
    logger.info(f'Node info: rank {args.rank} of world size {args.world_size}')
    cfg.args = args
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # distributed logging and ignore warning message
    logging.setup_logging(cfg.LOGDIR)
    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'train_logs'))

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model
    model = build_model(cfg)
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        print(name, param.requires_grad)\

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
    #                                                   output_device=args.local_rank, find_unused_parameters=True)
    optimizer = construct_optimizer(model, cfg)
    algo = get_algo(cfg)
    train_dataset = Meview(cfg)
    train_dataset.create_data(0)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_dataset = Meview(cfg)
    val_dataset.select_data(0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                            shuffle=True, drop_last=True)

    """Trains model and evaluates on relevant downstream tasks."""
    start_epoch = load_checkpoint(cfg, model, optimizer)
    cfg.TRAIN.MAX_ITERS = cfg.TRAIN.MAX_EPOCHS * len(train_loader)
    scheduler = construct_scheduler(optimizer, cfg)
    for cur_epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCHS -cfg.TRAIN.MAX_EPOCHS+start_epoch+10):
        logger.info(
            f"Traning epoch {cur_epoch}/{cfg.TRAIN.MAX_EPOCHS}, {len(train_loader)} iters each epoch")
        train(cfg, train_loader, model, optimizer,
              scheduler, algo, cur_epoch, summary_writer)
        val(cfg, val_loader, model, algo, cur_epoch, summary_writer)
        if du.is_root_proc() and ((cur_epoch+1) % cfg.CHECKPOINT.SAVE_INTERVAL == 0 or cur_epoch == cfg.TRAIN.MAX_EPOCHS-1):
            save_checkpoint(cfg, model, optimizer, cur_epoch)
        du.synchronize()
    # torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()