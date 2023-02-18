# coding=utf-8

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
    def __init__(self, cfg, trainID):
        super().__init__()
        self.id = trainID
        self.cfg = cfg
        self.peroid = 15
        self.sliding = 1
        self.TrainVal_thresh = 0.8

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
            p_data, p_label, n_data, n_label = self.to_PosNeg_data(
                images, labels)
            pos_data = torch.cat((pos_data, p_data))
            pos_label = torch.cat((pos_label, p_label))
            neg_data = torch.cat((neg_data, n_data))
            neg_label = torch.cat((neg_label, n_label))
        self.pos_data = pos_data.type(torch.float32) / 255.0
        self.pos_label = pos_label.type(torch.long)
        self.neg_data = neg_data.type(torch.float32) / 255.0
        self.neg_label = neg_label.type(torch.long)

    def generate_splitList(self):
        mask = torch.rand(len(self.pos_data))
        mask[mask > self.TrainVal_thresh] = True
        mask[mask <= self.TrainVal_thresh] = False
        self.splitList = mask

    def to_PosNeg_data(self, data, label):
        pos_data, neg_data = [], []
        pos_label, neg_label = [], []
        for idx in range(len(label)):
            if sum(label[idx:idx+self.peroid]) > 0:
                pos_data.append(data[idx:idx+self.peroid])
                pos_label.append(data[idx:idx+self.peroid])
            else:
                neg_data.append(data[idx:idx+self.peroid])
                neg_label.append(data[idx:idx+self.peroid])
        return torch.stack(pos_data), torch.stack(pos_label), torch.stack(neg_data), torch.stack(neg_label)

    def get_trainingDataset(self):
        mask = not self.splitList
        return Unduplicated_SequenceDataset(self.pos_data[mask], self.pos_label[mask], self.neg_data[mask], self.neg_label[mask])

    def get_valDataset(self):
        mask = self.splitList
        data = torch.cat((self.pos_data[mask], self.neg_data[mask]))
        label = torch.cat((self.pos_label[mask], self.neg_label[mask]))
        return SequenceDataset(data, label)

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
        return SequenceDataset(sequence_data, sequence_label)


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label
        self.mask = torch.Tensor([1 for _ in range(self.peroid)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.mask


class Unduplicated_SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, pos_data, neg_data, pos_label, neg_label):
        super().__init__(self)
        self.iteral = 0
        self.minlen = min(len(pos_data), len(neg_data))*2
        if len(pos_data) < len(neg_data):
            self.data = torch.cat((pos_data, neg_data))
            self.label = torch.cat((pos_label, neg_label))
        else:
            self.data = torch.cat((neg_data, pos_data))
            self.label = torch.cat((neg_label, pos_label))
        self.mask = torch.Tensor([1 for _ in range(self.peroid)])

    def create_rmList(self):
        minlen = self.minlen
        maxlen = len(self.data) - minlen
        self.rmMaxList = random.sample(range(minlen, len(self.data)), maxlen)
        self.rmBatchList = list(range(minlen)) + self.rmMaxList[:minlen]
        del self.rmMaxList[:minlen]

    def __len__(self):
        minlen = self.minlen
        maxlen = len(self.data) - minlen
        return int(maxlen / minlen) * minlen * 2

    def __getitem__(self, index):
        if len(self.rmBatchList) == 0:
            self.rmBatchList = list(range(self.minlen)) + \
                self.rmMaxList[:self.minlen]
            del self.rmMaxList[:self.minlen]
        index = index % len(self.rmBatchList)
        value = self.rmBatchList[index]
        del self.rmBatchList[index]
        return self.data[value], self.label[value], self.mask
