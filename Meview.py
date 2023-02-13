# coding=utf-8

import re
import glob
import random
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
    paths = sorted(glob.glob(f'{root}/*{file_type}'), key=natural_keys)
    return paths


class Meview(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.peroid = 15
        self.sliding = 2
        self.thresh = 0.8

    def load_validation_data(self, assignID):
        subject = SUBJECTS[assignID]
        input_paths = get_file_paths(
            f'{self.cfg.PATH_TO_DATASET}/{subject}', '.png')
        images = torch.stack([read_image(path) for path in input_paths])
        labels = torch.Tensor(
            [1 if ONSET[assignID] <= i < OFFSET[assignID] else 0 for i in range(len(images))])
        self.train_data = images.unfold(
            0, self.peroid, self.sliding).type(torch.float32) / 255.0
        self.train_data = self.train_data.permute(0, 4, 1, 2, 3)
        self.train_label = labels.unfold(
            0, self.peroid, self.sliding).type(torch.long)

    def under_sampling(self, sid, images, labels):
        last_images, last_labels = [], []
        rmList = torch.rand(NUM_FRAMES[sid]-self.peroid)
        for idx, value in enumerate(rmList):
            # including positive in peroid
            if idx < OFFSET[sid] and idx+self.peroid > ONSET[sid]:
                pass
            # random selected
            elif value > self.thresh:
                pass
            else:
                continue
            last_images.append(images[idx:idx+self.peroid])
            last_labels.append(labels[idx:idx+self.peroid])
        return torch.stack(last_images), torch.stack(last_labels)

    # total data including train/val
    def load_data(self, cfg):
        train_data, train_label = torch.Tensor(), torch.Tensor()
        for sid, subject in enumerate(SUBJECTS):
            print(f"start to load data from {SUBJECTS[sid]}")
            input_paths = get_file_paths(
                f'{cfg.PATH_TO_DATASET}/{subject}', '.png')
            images = torch.stack([read_image(path) for path in input_paths])
            labels = torch.Tensor(
                [1 if ONSET[sid] <= i < OFFSET[sid] else 0 for i in range(len(images))])
            train_data = torch.cat((train_data, images))
            train_label = torch.cat((train_label, labels))
        self.train_data = train_data.type(torch.float32) / 255.0
        self.train_label = train_label.type(torch.long)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        return self.train_data[index], self.train_label[index], torch.Tensor([1 for _ in range(self.peroid)])


class CheatedMeview(Meview):
    def __init__(self, cfg):
        super().__init__(cfg)

    def load_validation_data(self, assignID):
        subject = SUBJECTS[assignID]
        input_paths = get_file_paths(
            f'{self.cfg.PATH_TO_DATASET}/{subject}', '.png')
        images = torch.stack([self.toCheat2(read_image(path)) if ONSET[assignID] <= i < OFFSET[assignID] else self.toCheat(
            read_image(path)) for i, path in enumerate(input_paths)])
        labels = torch.Tensor(
            [1 if ONSET[assignID] <= i < OFFSET[assignID] else 0 for i in range(len(images))])
        self.train_data = images.unfold(
            0, self.peroid, self.sliding).type(torch.float32) / 255.0
        self.train_data = self.train_data.permute(0, 4, 1, 2, 3)
        self.train_label = labels.unfold(
            0, self.peroid, self.sliding).type(torch.long)

    # format-> batch_size, num_steps, c, h, w = x.shape
    def load_traning_data(self, exceptID=-1):
        train_data, train_label = torch.Tensor(), torch.Tensor()
        for sid, subject in enumerate(SUBJECTS):
            if sid == exceptID:
                continue
            input_paths = get_file_paths(
                f'{self.cfg.PATH_TO_DATASET}/{subject}', '.png')
            images = torch.stack([self.toCheat2(read_image(path)) if ONSET[sid] <= i < OFFSET[sid] else self.toCheat(
                read_image(path)) for i, path in enumerate(input_paths)])
            labels = torch.Tensor(
                [1 if ONSET[sid] <= i < OFFSET[sid] else 0 for i in range(len(images))])
            images, labels = self.under_sampling(sid, images, labels)
            train_data = torch.cat((train_data, images))
            train_label = torch.cat((train_label, labels))
        self.train_data = train_data.type(torch.float32) / 255.0
        self.train_label = train_label.type(torch.long)

    def __len__(self):
        return len(self.itemList)

    def __getitem__(self, index):
        index = self.itemList[index]
        data = self.train_data[index:index+self.peroid]
        label = self.train_label[index:index+self.peroid]
        return data, label, torch.Tensor([1 for _ in range(self.peroid)])


if __name__ == '__main__':
    from utils.parser import parse_args, load_config

    args = parse_args()
    cfg = load_config(args)
    dataset = CheatedMeview(cfg)
    dataset.load_traning_data(1)
    print(dataset.train_data.shape)
    print(dataset.train_label.shape)
    dataset.load_cheatedTraning_data(1)
    print(dataset.train_data.shape)
    print(dataset.train_label.shape)
