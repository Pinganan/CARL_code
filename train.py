


import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import carl_model
from algos import get_algo
from config import get_cfg
from dataset import MeviewDataLoader
from optimizer import construct_optimizer


SUBJECTS = ["sub01_01", "sub02_01", "sub03_01", "sub05_02", "sub06_01",
            "sub07_05", "sub07_09", "sub07_10", "sub08_01", "sub10_01",
            "sub11_03", "sub11_05", "sub13_01", "sub14_01", "sub14_02",
            "sub15_01", "sub15_02", "sub16_02", "sub16_03"]


def cal_evaluation(groundTruth, pred):
    tn, fp, fn, tp = confusion_matrix(groundTruth, pred, labels=[0, 1]).ravel()
    accuracy = (tn+tp) / (tn+fp+fn+tp) if (tn+fp+fn+tp) != 0 else 0
    precision = tp / (fp+tp) if (fp+tp) != 0 else 0
    recall = tp / (tp+fn) if (tp+fn) != 0 else 0
    f1_score = (2*precision*recall) / (precision+recall) if (precision+recall) != 0 else 0
    return tp, tn, fp, fn, accuracy, f1_score, recall


def mk_dir(path):
    index = path.rfind('/')
    if "." in path[index:]:
        os.makedirs(path[:index], exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)


def classVectortoString(inputs):
    out = ""
    for inp in inputs:
        out += "|" if inp==1 else "_"
    return out


def train(cfg, train_loader, model, optimizer, algo, cur_epoch, tf_writer, txt_writer):
    global trainSubjectID
    tmp_subject = SUBJECTS[trainSubjectID]
    groundTruth, predicts = [], []

    model.train()
    train_loader = tqdm(train_loader, total=len(train_loader), desc=f'training {tmp_subject}')
    for batch_size, (videosSet, labelsSet, masksSet) in enumerate(train_loader):
        for batch_step in range(len(videosSet)):
            optimizer.zero_grad()
            videos = videosSet[batch_step].cuda()
            labels = labelsSet[batch_step].cuda()
            masks = masksSet[batch_step].cuda()
            loss_dict, gts, preds = algo.compute_loss(model, videos, labels, masks)
            loss = loss_dict["loss"]
            loss.backward()

            # Update the parameters.
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZER.GRAD_CLIP)
            optimizer.step()
            for key in loss_dict:
                loss_dict[key][torch.isnan(loss_dict[key])] = 0

            groundTruth += gts.cpu()
            predicts += preds.cpu()
        # Record Loss into tensorboard
        tf_writer.add_scalar(f'{tmp_subject} / Train - loss', loss.item(), cur_epoch*len(train_loader)+batch_size)
    # Record each subject f1, recall, confusion matrix
    tp, tn, fp, fn, accuracy, f1_score, recall = cal_evaluation(groundTruth, predicts)
    txt_writer.write(f"\tTrain result\t{tp=:5d}, {tn=:5d}, {fp=:5d}, {fn=:5d}, {accuracy=:.3f}, {f1_score=:.3f}, {recall=:.3f}\n")


def val(val_loader, model, algo, cur_epoch, tf_writer, txt_writer):
    global trainSubjectID
    tmp_subject = SUBJECTS[trainSubjectID]
    groundTruth, predicts = [], []
    loss = None
    txt_writer.write(f"\tValid\n")

    model.eval()
    with torch.no_grad():
        for batch_step, (videos, labels, video_masks) in enumerate(val_loader):
            videos = videos.cuda()
            labels = labels.cuda()
            video_masks = video_masks.cuda()
            loss_dict, gts, preds = algo.compute_loss(model, videos, labels, video_masks, training=False)
            loss = loss_dict["loss"]

            gts = gts.cpu()
            preds = preds.cpu()
            groundTruth += gts
            predicts += preds
            txt_writer.write(f"\t\t  ground\t{classVectortoString(gts)}\n")
            txt_writer.write(f"\t\t  predic\t{classVectortoString(preds)}\n")
    tf_writer.add_scalar(f'{tmp_subject} / Valid - loss', loss.item(), cur_epoch)
    # Record each subject f1, recall, confusion matrix
    tp, tn, fp, fn, accuracy, f1_score, recall = cal_evaluation(groundTruth, predicts)
    txt_writer.write(f"\tValid result\t{tp=:5d}, {tn=:5d}, {fp=:5d}, {fn=:5d}, {accuracy=:.3f}, {f1_score=:.3f}, {recall=:.3f}\n")


def test(test_loader, model, algo, cur_epoch, tf_writer, txt_writer):
    global trainSubjectID
    tmp_subject = SUBJECTS[trainSubjectID]
    groundTruth, predicts = [], []
    loss = None
    txt_writer.write(f"\tTest\n")
    model.eval()
    with torch.no_grad():
        for batch_step, (videos, labels, video_masks) in enumerate(test_loader):
            videos = videos.cuda()
            labels = labels.cuda()
            video_masks = video_masks.cuda()
            loss_dict, gts, preds = algo.compute_loss(model, videos, labels, video_masks, training=False)
            loss = loss_dict["loss"]

            gts = gts.cpu()
            preds = preds.cpu()
            groundTruth += gts
            predicts += preds
            txt_writer.write(f"\t\t  ground\t{classVectortoString(gts)}\n")
            txt_writer.write(f"\t\t  predic\t{classVectortoString(preds)}\n")
    tf_writer.add_scalar(f'{tmp_subject} / Test - loss', loss.item(), cur_epoch)
    # Record each subject f1, recall, confusion matrix
    tp, tn, fp, fn, accuracy, f1_score, recall = cal_evaluation(groundTruth, predicts)
    txt_writer.write(f"\tTest  result\t{tp=:5d}, {tn=:5d}, {fp=:5d}, {fn=:5d}, {accuracy=:.3f}, {f1_score=:.3f}, {recall=:.3f}\n")


def main(cfg, trainSubjectID, tf_writer, txt_writer):
    algo = get_algo(cfg)
    model = carl_model.TransformerModel(cfg)
    optimizer = construct_optimizer(model, cfg)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    loader = MeviewDataLoader(cfg=cfg, trainID=trainSubjectID)
    train_loader = loader.get_trainLoader()
    val_loader = loader.get_valLoader()
    test_loader = loader.get_testLoader()

    """Trains model and evaluates on relevant downstream tasks."""
    print(f"Start to Train {SUBJECTS[trainSubjectID]}")
    txt_writer.write(f"{SUBJECTS[trainSubjectID]}:\n")
    for cur_epoch in range(10):
        train(cfg, train_loader, model, optimizer, algo, cur_epoch, tf_writer, txt_writer)
        val(val_loader, model, algo, cur_epoch, tf_writer, txt_writer)
        torch.save(model.state_dict(), f"./_STATES/{cfg.SAVE_PATH}/{SUBJECTS[trainSubjectID]}_{cur_epoch}.pth")
        test(test_loader, model, algo, cur_epoch, tf_writer, txt_writer)
        txt_writer.write(f"\n")
    print(f"Finish {SUBJECTS[trainSubjectID]}")


if __name__ == '__main__':
    cfg = get_cfg()
    PATH = cfg.SAVE_PATH
    mk_dir(f"./_STATES/{PATH}/")
    mk_dir(f"./_BOARDS/{PATH}/")
    mk_dir(f"./_LOGS/{PATH}/")
    
    torch.manual_seed(1015)
    
    tf_writer = SummaryWriter(log_dir=f"_BOARDS/{PATH}/")
    for trainSubjectID in range(len(SUBJECTS)):
        txt_writer = open(f"./_LOGS/{PATH}/{SUBJECTS[trainSubjectID]}.txt", "w")
        main(cfg, trainSubjectID, tf_writer, txt_writer)
        txt_writer.close()
    tf_writer.close()
