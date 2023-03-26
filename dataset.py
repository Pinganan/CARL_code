

import re
import glob
import torch
import random
from torchvision import transforms
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
        self.is_overlap = config.DATA.OVERLAP
        self.is_smooth = config.LABEL.SMOOTH
        self.rsize = transforms.Resize(size=config.DATA.RESIZE)
        self.data, self.label = list(), list()
        self.train_idx, self.val_idx = list(), list()
        self.load_inner_data()
        self.load_outer_data()

    def load_inner_data(self):
        all_idx = list()
        for sub_idx, subject in enumerate(SUBJECTS):
            print(f"Start loading {subject} data")
            if sub_idx == self.id:
                continue
            data_paths = get_file_paths(f"{self.root}/{subject}", ".png")
            assert len(data_paths) == NUM_FRAMES[sub_idx], f"{subject=}: {len(data_paths)=}, {NUM_FRAMES[sub_idx]=}"

            images = [self.rsize(read_image(path))/255.0 for path in data_paths]
            start = max(0, (ONSET[sub_idx]-self.label_bias))
            end = min(NUM_FRAMES[sub_idx], (OFFSET[sub_idx] + self.label_bias))
            if self.is_smooth:
                length = end - start + 1
                mid = int((start+end) / 2)
                labels = torch.Tensor([abs(i-mid) / length if i in range(start, end) else 0 for i in range(NUM_FRAMES[sub_idx])])
            else:
                labels = torch.Tensor([1 if i in range(start, end) else 0 for i in range(NUM_FRAMES[sub_idx])])
            all_idx += [x for x in range(len(self.data), len(self.data)+len(labels)-self.peroid)]
            self.data += images
            self.label += labels
            if sub_idx == 2:
                break
        self.train_idx, self.val_idx = self.generate_trainval_idx(all_idx)

    def load_outer_data(self):
        subject = SUBJECTS[self.id]
        data_paths = get_file_paths(f"{self.root}/{subject}", ".png")
        assert len(data_paths) == NUM_FRAMES[self.id], f"{subject=}: {len(data_paths)=}, {NUM_FRAMES[self.id]=}"
        
        images = [self.rsize(read_image(path))/255.0 for path in data_paths]
        start = max(0, (ONSET[self.id]-self.label_bias))
        end = min(NUM_FRAMES[self.id], (OFFSET[self.id] + self.label_bias))
        if self.is_smooth:
            length = end - start + 1
            mid = int((start+end) / 2)
            labels = torch.Tensor([abs(i-mid) / length if i in range(start, end) else 0 for i in range(NUM_FRAMES[self.id])])
        else:
            labels = torch.Tensor([1 if i in range(start, end) else 0 for i in range(NUM_FRAMES[self.id])])
        self.test_data = images
        self.test_label = labels
        self.test_idx = [i for i in range(0, len(self.test_data)-self.peroid, self.stride)]

    def generate_trainval_idx(self, all_idx):
        train_amount = int(self.split_rate * len(all_idx))
        random.shuffle(all_idx)
        train_idx = set(all_idx[:train_amount])
        val_idx = all_idx[train_amount:]
        if not self.is_overlap:
            tmp = set()
            for idx in val_idx:
                tmp |= set([x for x in range(idx, idx+self.peroid)])
            train_idx -= tmp
        train_idx = list(train_idx)
        random.shuffle(train_idx)
        return train_idx, val_idx


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, config, data_idx, data, label):
        super().__init__()
        self.config = config
        self.data_idx = data_idx
        self.data = data
        self.label = label
        self.peroid = config.DATA.PEROID

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, index):
        start = self.data_idx[index]
        end = start + self.peroid
        data = torch.stack(self.data[start:end])
        label = torch.Tensor(self.label[start:end])
        mask = torch.ones(label.shape)
        return data, label, mask


class UnbalanceSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data_idx, data, label):
        super().__init__()
        self.peroid = cfg.DATA.PEROID
        self.inner_batch = cfg.DATA.INNER_BATCH
        self.data = data
        self.label = label
        self.OverSample(data_idx)
        self.fit_batch()

    def split_MoreLess(self, data_idx):
        pos_idx, neg_idx = list(), list()
        for idx in data_idx:
            if sum(self.label[idx:idx+self.peroid]) > 0:
                pos_idx.append(idx)
            else:
                neg_idx.append(idx)
        if len(pos_idx) > len(neg_idx):
            return neg_idx, pos_idx
        else:
            return pos_idx, neg_idx

    def OverSample(self, data):
        more_idx, less_idx = self.split_MoreLess(data)
        # random.shuffle(more_idx)
        # random.shuffle(less_idx)
        self.more_idx = more_idx
        self.less_idx = less_idx
        self.inner_batch = min(int(self.inner_batch/2), len(less_idx))

    def fit_batch(self):
        vacancy_len = -len(self.more_idx) % self.inner_batch
        self.more_idx += self.more_idx[:vacancy_len]
        vacancy_len = -len(self.less_idx) % self.inner_batch
        self.less_idx += self.less_idx[:vacancy_len]

    def __len__(self):
        return int(len(self.more_idx) * len(self.less_idx) / self.inner_batch**2)

    def __getitem__(self, idx):
        data = list()
        label = list()
        s = int(idx % (len(self.less_idx) / self.inner_batch) * self.inner_batch)
        for idx in self.less_idx[s:s+self.inner_batch]:
            tmp_d = torch.stack(self.data[idx:idx+self.peroid])
            tmp_l = torch.Tensor(self.label[idx:idx+self.peroid])
            data.append(tmp_d)
            label.append(tmp_l)

        s = int(idx % (len(self.more_idx) / self.inner_batch) * self.inner_batch)
        for idx in self.more_idx[s:s+self.inner_batch]:
            tmp_d = torch.stack(self.data[idx:idx+self.peroid])
            tmp_l = torch.Tensor(self.label[idx:idx+self.peroid])
            data.append(tmp_d)
            label.append(tmp_l)
        data = torch.stack(data).float()
        label = torch.stack(label).long()
        mask = torch.ones(label.shape)
        return data, label, mask


class MeviewDataLoader():
    def __init__(self, cfg, trainID):
        self.cfg = cfg
        self.dataset = MeviewDataset(cfg, trainID)
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.train_worker = cfg.TRAIN.NUM_WORKER

    def get_trainLoader(self):
        dataset = UnbalanceSequenceDataset(
            self.cfg, self.dataset.train_idx, self.dataset.data, self.dataset.label)
        # for i in range(dataset.__len__()):
        #     d, l, mask = dataset.__getitem__(i)
        #     print(d.shape)
        #     print(l.shape)
        #     print(mask.shape)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.train_worker)

    def get_testLoader(self):
        dataset = SequenceDataset(
            self.cfg, self.dataset.test_idx, self.dataset.test_data, self.dataset.test_label)
        # d, l, mask = dataset.__getitem__(0)
        # print(d.shape)
        # print(l.shape)
        # print(mask.shape)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.train_worker)

    def get_valLoader(self):
        dataset = SequenceDataset(
            self.cfg, self.dataset.val_idx, self.dataset.data, self.dataset.label)
        # d, l, mask = dataset.__getitem__(0)
        # print(d.shape)
        # print(l.shape)
        # print(mask.shape)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.train_worker)


if __name__ == '__main__':
    import config
    cfg = config.get_cfg()

    dataset = MeviewDataset(cfg, 1)
    print(type(dataset.data), len(dataset.data), type(
        dataset.data[0]), dataset.data[0].shape)
    print(type(dataset.label), len(dataset.label), type(
        dataset.label[0]), dataset.label[0].shape)

    print(dataset.train_idx)
    print(dataset.val_idx)

    # loader = MeviewDataLoader(cfg, 1)
    # train_loader = loader.get_trainLoader()
    # test_loader = loader.get_testLoader()
    # val_loader = loader.get_valLoader()
