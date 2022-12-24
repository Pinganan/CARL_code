# coding=utf-8

import torch
import re
import cv2
import glob
import numpy as np
from utils.parser import parse_args, load_config
from torchvision import transforms


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


# batch_size, num_steps, c, h, w = x.shape
class Meview(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_data = []
        self.train_label = []
        self.create_data()

    def create_data(self):
        for sid, subject in enumerate(SUBJECTS):
            inputs = get_file_paths(
                f'{self.cfg.PATH_TO_DATASET}/{subject}', '.png')
            images = [cv2.imread(p) for p in inputs]

            for i in range(len(images) - 30):
                self.train_data.append(images[i:i+30])
                self.train_label.append(
                    [1 if ONSET[sid] <= i < OFFSET[sid] else 0 for i in range(i, i+15)])
        self.train_data = torch.Tensor(np.array(self.train_data))
        self.train_label = torch.Tensor(np.array(self.train_label))
        batch, num_frames, height, width, channel = self.train_data.shape
        self.train_data = self.train_data.reshape(
            (batch, num_frames, channel, height, width))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        return self.train_data[index], self.train_label[index], torch.Tensor([1 for _ in range(15)])


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)
    dataset = Meview(cfg)
    print(dataset.train_data[0].shape)
    print(dataset.train_label[0].shape)
