
import re
import glob
import random
from enum import Enum
import torch
from torchvision.io import read_image


SUBJECTS = ["sub01_01", "sub02_01", "sub03_01", "sub05_02", "sub06_01",
            "sub07_05", "sub07_09", "sub07_10", "sub08_01", "sub10_01",
            "sub11_03", "sub11_05", "sub13_01", "sub14_01", "sub14_02",
            "sub15_01", "sub15_02", "sub16_02", "sub16_03"]
ONSET = [48, 87, 77, 70, 15, 60, 72, 22, 12,
         10, 51, 4, 14, 31, 32, 37, 123, 43, 1]
OFFSET = [61, 98, 90, 82, 26, 76, 88, 33, 21,
          20, 66, 10, 32, 48, 45, 44, 135, 59, 16]
NUM_FRAMES = [89, 117, 138, 107, 60, 76, 99, 126,
              108, 191, 69, 58, 63, 63, 63, 62, 168, 61, 47]


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_file_paths(root, file_type="/"):
    return sorted(glob.glob(f'{root}/*{file_type}'), key=natural_keys)


class MeviewDataset(object):
    def __init__(self, cfg, trainID, peroid=15, sliding=5):
        super().__init__()
        self.id = trainID
        self.cfg = cfg
        self.peroid = peroid
        self.sliding = sliding
        self.load_sequence_image()
        self.split_TrainVal_data()

    def load_sequence_image(self):
        pos_data, pos_label = torch.Tensor(), torch.Tensor()
        neg_data, neg_label = torch.Tensor(), torch.Tensor()
        for sid, subject in enumerate(SUBJECTS):
            if sid == self.id:
                continue
            input_paths = get_file_paths(
                f'{self.cfg.PATH_TO_DATASET}/{subject}', '.png')
            images = torch.stack([read_image(path) for path in input_paths])
            labels = torch.Tensor(
                [1 if ONSET[sid] <= i < OFFSET[sid] else 0 for i in range(len(images))])
            p_data, p_label, n_data, n_label = self.split_PosNeg_data(images, labels)
            pos_data = torch.cat((pos_data, p_data))
            pos_label = torch.cat((pos_label, p_label))
            neg_data = torch.cat((neg_data, n_data))
            neg_label = torch.cat((neg_label, n_label))
            # print(f"Loading {SUBJECTS[sid]} data finished")
        self.pos_data = pos_data.type(torch.float32) / 255.0
        self.pos_label = pos_label.type(torch.long)
        self.neg_data = neg_data.type(torch.float32) / 255.0
        self.neg_label = neg_label.type(torch.long)

    def split_PosNeg_data(self, data, label):
        pos_data, neg_data = [], []
        pos_label, neg_label = [], []
        for idx in range(0, len(label)-self.peroid, self.sliding):
            if sum(label[idx:idx+self.peroid]) > 0:
                pos_data.append(data[idx:idx+self.peroid])
                pos_label.append(label[idx:idx+self.peroid])
            else:
                neg_data.append(data[idx:idx+self.peroid])
                neg_label.append(label[idx:idx+self.peroid])
        return torch.stack(pos_data), torch.stack(pos_label), torch.stack(neg_data), torch.stack(neg_label)

    def split_TrainVal_data(self, TrainVal_thresh=0.8):
        minlen = min(len(self.pos_data), len(self.neg_data))
        thresh = torch.rand(len(self.pos_data))
        thresh = thresh < TrainVal_thresh

        train_pos_data = self.pos_data[:minlen][thresh]
        train_pos_data = torch.cat((train_pos_data, self.pos_data[minlen:]))
        train_pos_lab = self.pos_label[:minlen][thresh]
        train_pos_lab = torch.cat((train_pos_lab, self.pos_label[minlen:]))
        train_neg_data = self.neg_data[:minlen][thresh]
        train_neg_data = torch.cat((train_neg_data, self.neg_data[minlen:]))
        train_neg_lab = self.neg_label[:minlen][thresh]
        train_neg_lab = torch.cat((train_neg_lab, self.neg_label[minlen:]))
        
        self.train_dataset = Unduplicated_Dataset(train_pos_data, train_pos_lab,
                                                  train_neg_data, train_neg_lab,
                                                  peroid=self.peroid, MaxBatch=10)

        val_pos_data = self.pos_data[:minlen][~thresh]
        val_neg_data = self.neg_data[:minlen][~thresh]
        val_pos_lab = self.pos_label[:minlen][~thresh]
        val_neg_lab = self.neg_label[:minlen][~thresh]

        self.val_dataset = SequenceDataset(torch.cat((val_pos_data, val_neg_data)),
                                           torch.cat((val_pos_lab, val_neg_lab)),
                                           peroid=self.peroid)

    def get_valDataset(self):
        return self.val_dataset

    def get_trainDataset(self):
        return self.train_dataset

    def get_testingDataset(self):
        del self.pos_data
        del self.pos_label
        del self.neg_data
        del self.neg_label
        subject = SUBJECTS[self.id]
        input_paths = get_file_paths(
            f'{self.cfg.PATH_TO_DATASET}/{subject}', '.png')
        images = torch.stack([read_image(path) for path in input_paths])
        labels = torch.Tensor(
            [1 if ONSET[self.id] <= i < OFFSET[self.id] else 0 for i in range(len(images))])
        sequence_data = images.unfold(
            0, self.peroid, self.sliding).type(torch.float32) / 255.0
        sequence_data = sequence_data.permute(0, 4, 1, 2, 3)
        sequence_label = labels.unfold(
            0, self.peroid, self.sliding).type(torch.long)
        return SequenceDataset(sequence_data, sequence_label, self.peroid)


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, peroid=15):
        super().__init__()
        self.data = data
        self.label = label
        self.mask = torch.ones(peroid)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.mask


class Unduplicated_Dataset(torch.utils.data.Dataset):
    def __init__(self, pdata, plabel, ndata, nlabel, peroid=15, MaxBatch=10):
        super().__init__()
        self.innerBatch = int(min(MaxBatch/2, len(pdata), len(ndata))) * 2
        self.pdata, self.plabel = self.block_data(pdata, plabel)
        self.ndata, self.nlabel = self.block_data(ndata, nlabel)
        self.mask = torch.ones(self.innerBatch, peroid)

    def block_data(self, data, label):
        batch = int(self.innerBatch / 2)
        ranlist = torch.randperm(len(data))
        data = data[ranlist]
        data = data[len(data) % batch:]
        label = label[ranlist]
        label = label[len(label) % batch:]
        return torch.split(data, batch), torch.split(label, batch)

    def __len__(self):
        return len(self.pdata) * len(self.ndata)

    def __getitem__(self, index):
        return torch.cat((self.pdata[index % len(self.pdata)], self.ndata[index % len(self.ndata)])), \
            torch.cat((self.plabel[index % len(self.plabel)], self.nlabel[index % len(self.nlabel)])), \
            self.mask
