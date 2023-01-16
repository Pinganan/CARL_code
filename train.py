# coding=utf-8

import os
import torch
from tqdm import tqdm
from datetime import datetime
from utils.parser import parse_args, load_config
from models import build_model, load_checkpoint
from utils.optimizer import construct_optimizer, construct_scheduler
from Meview import Meview
from torch.utils.data import DataLoader
from algos import get_algo
from sklearn.metrics import f1_score, recall_score, confusion_matrix


SUBJECTS = ["sub01_01", "sub02_01", "sub03_01", "sub05_02", "sub06_01",
            "sub07_05", "sub07_09", "sub07_10", "sub08_01", "sub10_01",
            "sub11_03", "sub11_05", "sub13_01", "sub14_01", "sub14_02",
            "sub15_01", "sub15_02", "sub16_02", "sub16_03"]


def classify_evaluation(gt, pred):
    gt = gt.cpu()
    pred = pred.cpu()
    f1 = f1_score(gt, pred, zero_division=1)
    recall = recall_score(gt, pred, zero_division=1)
    return f1, recall


def train(cfg, train_loader, model, optimizer, scheduler, algo, cur_epoch, writer):
    global trainSubjectID
    model.train()
    optimizer.zero_grad()
    total_loss = {}
    train_loader = tqdm(train_loader, total=len(train_loader),
                        desc=f'[train-{SUBJECTS[trainSubjectID]}] loss=0.000, f1=0.000, recall=0.000')

    groundTruth, predicts = torch.Tensor(), torch.Tensor()
    for videos, _labels, video_masks in train_loader:
        optimizer.zero_grad()
        loss_dict, gts, preds = algo.compute_loss(
            model, videos, _labels, video_masks)
        loss = loss_dict["loss"]
        loss.backward()

        groundTruth = torch.cat((groundTruth.cuda(), gts.cuda()))
        predicts = torch.cat((predicts.cuda(), preds.cuda()))
        f1, recall = classify_evaluation(groundTruth, predicts)
        train_loader.set_description(
            f"[train-{SUBJECTS[trainSubjectID]}] {loss=:.3f}, {f1=:.3f}, {recall=:.3f}")
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

    f1, recall = classify_evaluation(groundTruth.cuda(), predicts.cuda())
    tn, fp, fn, tp = confusion_matrix(
        groundTruth.cpu(), predicts.cpu(), labels=[0, 1]).ravel()
    writer.write(
        f"[train results] {tp=:5d}, {tn=:5d}, {fp=:5d}, {fn=:5d}, {f1=:.3f}, {recall=:.3f}\n")

    if cur_epoch != cfg.TRAIN.MAX_EPOCHS-1:
        scheduler.step()


def val(val_loader, model, algo, writer):
    model.eval()
    total_loss = {}

    groundTruth, predicts = torch.Tensor(), torch.Tensor()
    with torch.no_grad():
        for videos, labels, video_masks in val_loader:
            loss_dict, gts, preds = algo.compute_loss(
                model, videos, labels, video_masks, training=False)

            groundTruth = torch.cat((groundTruth.cuda(), gts.cuda()))
            predicts = torch.cat((predicts.cuda(), preds.cuda()))
            for key in loss_dict:
                loss_dict[key][torch.isnan(loss_dict[key])] = 0
                if key not in total_loss:
                    total_loss[key] = 0

    f1, recall = classify_evaluation(groundTruth.cuda(), predicts.cuda())
    tn, fp, fn, tp = confusion_matrix(
        groundTruth.cpu(), predicts.cpu(), labels=[0, 1]).ravel()
    writer.write(
        f"[-val- results] {tp=:5d}, {tn=:5d}, {fp=:5d}, {fn=:5d}, {f1=:.3f}, {recall=:.3f}\n")


def main(trainSubjectID, writer):
    args = parse_args()
    cfg = load_config(args)

    algo = get_algo(cfg)
    optimizer = construct_optimizer(model, cfg)
    scheduler = construct_scheduler(optimizer, cfg)
    model = build_model(cfg)
    load_checkpoint(cfg, model, optimizer)
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    train_dataset = Meview(cfg)
    train_dataset.create_data(trainSubjectID)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=True)
    val_dataset = Meview(cfg)
    val_dataset.select_data(trainSubjectID)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE)

    """Trains model and evaluates on relevant downstream tasks."""
    for cur_epoch in range(3):
        train(cfg, train_loader, model, optimizer,
              scheduler, algo, cur_epoch, writer)
        val(val_loader, model, algo, writer)
        break


if __name__ == '__main__':
    for trainSubjectID in range(19):
        now = datetime.now()
        writer = open(
            f"./train_logs/{now.year}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{SUBJECTS[trainSubjectID]}.txt", "w")
        main(trainSubjectID, writer)
        writer.close()
