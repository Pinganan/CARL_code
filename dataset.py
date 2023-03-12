

import re
import glob
import torch
import random
from torchvision.io import read_image
from torch.utils.data import DataLoader


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
    def __init__(self, config, trainID):
        self.id = trainID
        self.cfg = config
        self.root = config.DATA.ROOT_PATH
        self.peroid = config.DATA.PEROID
        self.stride = config.DATA.STRIDE
        self.split_rate = config.DATA.SPLIT_RATE
        self.label_bias = config.DATA.LABEL_BIAS
        self.load_inner_data()
        self.load_outer_data()

    def load_inner_data(self):
        in_data, in_label = list(), list()
        for sub_idx, subject in enumerate(SUBJECTS):
            print(f"Start load {subject}")
            if sub_idx == self.id:  # one out
                continue
            start = max(0, (ONSET[sub_idx]-self.label_bias))
            end = min(NUM_FRAMES[sub_idx]-1, (OFFSET[sub_idx] + self.label_bias))
            data_paths = get_file_paths(f"{self.root}/{subject}", ".png")
            assert len(data_paths) == NUM_FRAMES[sub_idx], f"{subject=}: {len(data_paths)=}, {NUM_FRAMES[sub_idx]=}"
            
            images = torch.stack([read_image(path) for path in data_paths])
            labels = torch.Tensor([1 if i in range(start, end) else 0 for i in range(NUM_FRAMES[sub_idx])])
            seq_data = images.unfold(0, self.peroid, self.stride).type(torch.float32) / 255.0
            seq_data = seq_data.permute(0, 4, 1, 2, 3)
            seq_label = labels.unfold(0, self.peroid, self.stride).type(torch.long)
            in_data += seq_data
            in_label += seq_label
            if sub_idx == 3:    # for testing
                break
        self.train_data = in_data
        self.train_label = in_label
        
    def load_outer_data(self):
        out_data, out_label = list(), list()
        subject = SUBJECTS[self.id]
        start = max(0, (ONSET[self.id]-self.label_bias))
        end = min(NUM_FRAMES[self.id]-1, (OFFSET[self.id] + self.label_bias))
        data_paths = get_file_paths(f"{self.root}/{subject}", ".png")
        assert len(data_paths) == NUM_FRAMES[self.id], f"{subject=}: {len(data_paths)=}, {NUM_FRAMES[self.id]=}"
        
        images = torch.stack([read_image(path) for path in data_paths])
        labels = torch.Tensor([1 if i in range(start, end) else 0 for i in range(NUM_FRAMES[self.id])])
        seq_data = images.unfold(0, self.peroid, self.stride).type(torch.float32) / 255.0
        out_data += seq_data.permute(0, 4, 1, 2, 3)
        out_label += labels.unfold(0, self.peroid, self.stride).type(torch.long)

        self.test_data = out_data
        self.test_label = out_label


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, config, data, label):
        super().__init__()
        self.config = config
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], None


class UnbalanceSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data, label):
        super().__init__()
        self.inner_batch = cfg.DATA.INNER_BATCH
        self.OverSample(data, label)
        
    def split_MoreLess(self, data, label):
        pos_data, pos_label = list(), list()
        neg_data, neg_label = list(), list()
        for idx in range(len(data)):
            if sum(label[idx]) > 0:
                pos_data.append(data[idx])
                pos_label.append(label[idx])
            else:
                neg_data.append(data[idx])
                neg_label.append(label[idx])
        if len(neg_data) > len(pos_data):
            return neg_data, neg_label, pos_data, pos_label
        else:
            return pos_data, pos_label, neg_data, neg_label
    
    def OverSample(self, data, label):
        more_data, more_label, less_data, less_label = self.split_MoreLess(data, label)
        self.data = list()
        self.label = list()
        random.shuffle(more_data)
        random.shuffle(more_label)
        random.shuffle(less_data)
        random.shuffle(less_label)
        innerBatch = min(int(self.inner_batch/2), len(less_data))
        for less_idx in range(0, len(less_data), innerBatch):
            for more_idx in range(0, len(more_data), innerBatch):
                tmp_data = less_data[less_idx:less_idx+innerBatch] + more_data[more_idx:more_idx+innerBatch]
                tmp_label = less_label[less_idx:less_idx+innerBatch] + more_label[more_idx:more_idx+innerBatch]
                
                tmp_data = torch.stack(tmp_data)
                tmp_label = torch.stack(tmp_label)
                
                self.data.append(tmp_data)
                self.label.append(tmp_label)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class MeviewDataLoader():
    def __init__(self, cfg, trainID):
        self.cfg = cfg
        self.dataset = MeviewDataset(cfg, trainID)
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.train_worker = cfg.TRAIN.NUM_WORKER
    
    def get_trainLoader(self):
        dataset = UnbalanceSequenceDataset(self.cfg, self.dataset.train_data, self.dataset.train_label)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.train_worker)

    def get_testLoader(self):
        dataset = SequenceDataset(self.cfg, self.dataset.test_data, self.dataset.test_label)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.train_worker)


if __name__ == '__main__':
    import config
    cfg = config.get_cfg()
    
    dataset = MeviewDataset(cfg, 1)
    print(type(dataset.train_data), type(dataset.train_data[0]), len(dataset.train_data), dataset.train_data[0].shape)
    print(type(dataset.test_data), type(dataset.test_data[0]), len(dataset.test_data), dataset.test_data[0].shape)
    
    loader = MeviewDataLoader(cfg, 1)
    train_loader = loader.get_trainLoader()
    test_loader = loader.get_testLoader()
