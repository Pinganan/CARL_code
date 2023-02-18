# coding=utf-8

import torch
from tqdm import tqdm
from datetime import datetime
from utils.parser import parse_args, load_config
from models import build_model, load_checkpoint
from utils.optimizer import construct_optimizer, construct_scheduler
from Meview import Meview, CheatedMeview
from torch.utils.data import DataLoader
from algos import get_algo
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter


SUBJECTS = ["sub01_01", "sub02_01", "sub03_01", "sub05_02", "sub06_01",
            "sub07_05", "sub07_09", "sub07_10", "sub08_01", "sub10_01",
            "sub11_03", "sub11_05", "sub13_01", "sub14_01", "sub14_02",
            "sub15_01", "sub15_02", "sub16_02", "sub16_03"]


def classify_evaluation(gt, pred):
    f1 = f1_score(gt, pred, zero_division=1)
    recall = recall_score(gt, pred, zero_division=1)
    return f1, recall


def train(cfg, train_loader, model, optimizer, algo, cur_epoch, tf_writer, txt_writer):
    global trainSubjectID
    tmp_subject = SUBJECTS[trainSubjectID]
    groundTruth, predicts = [], []

    model.train()
    train_loader = tqdm(train_loader, total=len(train_loader),
                        desc=f'training {tmp_subject}')
    for batch_step, (videos, labels, video_masks) in enumerate(train_loader):
        optimizer.zero_grad()
        videos = videos.cuda()
        labels = labels.cuda()
        video_masks = video_masks.cuda()
        loss_dict, gts, preds = algo.compute_loss(
            model, videos, labels, video_masks)
        loss = loss_dict["loss"]
        loss.backward()

        # Update the parameters.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.OPTIMIZER.GRAD_CLIP)
        optimizer.step()
        for key in loss_dict:
            loss_dict[key][torch.isnan(loss_dict[key])] = 0

        # Record Loss into tensorboard
        tf_writer.add_scalar(f'{tmp_subject} / Train - loss',
                             loss.item(), cur_epoch*len(train_loader)+batch_step)

        groundTruth += gts.cpu()
        predicts += preds.cpu()
    # Record each subject f1, recall, confusion matrix
    f1, recall = classify_evaluation(groundTruth, predicts)
    tn, fp, fn, tp = confusion_matrix(
        groundTruth, predicts, labels=[0, 1]).ravel()
    txt_writer.write(
        f"\tTrain result\t{tp=:5d}, {tn=:5d}, {fp=:5d}, {fn=:5d}, {f1=:.3f}, {recall=:.3f}\n")
    print(
        f"train Pos/Neg = {sum(groundTruth)} / {len(groundTruth)-sum(groundTruth)}")


def val(val_loader, model, algo, cur_epoch, tf_writer, txt_writer):
    global trainSubjectID
    tmp_subject = SUBJECTS[trainSubjectID]
    groundTruth, predicts = [], []

    model.eval()
    with torch.no_grad():
        for batch_step, (videos, labels, video_masks) in enumerate(val_loader):
            videos = videos.cuda()
            labels = labels.cuda()
            video_masks = video_masks.cuda()
            loss_dict, gts, preds = algo.compute_loss(
                model, videos, labels, video_masks, training=False)
            loss = loss_dict["loss"]

            # Record Loss into tensorboard
            tf_writer.add_scalar(f'{tmp_subject} / Valid - loss',
                                 loss.item(), cur_epoch*len(val_loader)+batch_step)

            groundTruth += gts.cpu()
            predicts += preds.cpu()
        # Record each subject f1, recall, confusion matrix
        f1, recall = classify_evaluation(groundTruth, predicts)
        tn, fp, fn, tp = confusion_matrix(
            groundTruth, predicts, labels=[0, 1]).ravel()
        txt_writer.write(
            f"\tValid result\t{tp=:5d}, {tn=:5d}, {fp=:5d}, {fn=:5d}, {f1=:.3f}, {recall=:.3f}\n")
        print(
            f"valid Pos/Neg = {sum(groundTruth)} / {len(groundTruth)-sum(groundTruth)}")


def main(trainSubjectID, tf_writer, txt_writer):
    args = parse_args()
    cfg = load_config(args)
    algo = get_algo(cfg)
    model = build_model(cfg)
    optimizer = construct_optimizer(model, cfg)
    load_checkpoint(cfg, model, optimizer)
    # for name, param in model.named_parameters():
    #     if "classifier" in name:
    #         param.requires_grad = True
    #     elif "transformer" in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    train_dataset = Meview(cfg)
    train_dataset.load_traning_data(trainSubjectID)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    val_dataset = Meview(cfg)
    val_dataset.load_validation_data(trainSubjectID)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE)

    """Trains model and evaluates on relevant downstream tasks."""
    print(f"Start to Train {SUBJECTS[trainSubjectID]}")
    txt_writer.write(f"{SUBJECTS[trainSubjectID]}\n")
    for cur_epoch in range(5):
        train(cfg, train_loader, model, optimizer, algo, cur_epoch, tf_writer, txt_writer)
    val(val_loader, model, algo, cur_epoch, tf_writer, txt_writer)
    txt_writer.write(f"\n")
    print(f"Finish {SUBJECTS[trainSubjectID]}")


if __name__ == '__main__':
    now = datetime.now()
    path = f"{now.month:02d}_{now.day:02d}_{now.hour:02d}"
    tf_writer = SummaryWriter(log_dir=f"runs/{path}/")
    for trainSubjectID in range(len(SUBJECTS)):
        txt_writer = open(
            f"./train_logs/{path}_{SUBJECTS[trainSubjectID]}.txt", "w")
        main(trainSubjectID, tf_writer, txt_writer)
        txt_writer.close()
    tf_writer.close()
