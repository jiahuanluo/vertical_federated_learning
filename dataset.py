# encoding: utf-8

import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import random
import cv2
import pandas as pd
from sklearn.preprocessing import StandardScaler


class NUS_WIDE_2_Party:
    def __init__(self, data_dir, selected_labels_list, data_type, k):
        self.x = []  # the datapath of 2 different png files
        self.y = []  # the corresponding label
        self.data_dir = data_dir
        self.k = k
        self.selected_labels_list = selected_labels_list
        self.data_type = data_type
        Xa, Xb, y = self.find_class()
        scale_model = StandardScaler()
        self.Xa = scale_model.fit_transform(Xa)
        self.Xb = scale_model.fit_transform(Xb)
        y_ = []
        pos_count = 0
        neg_count = 0
        for i in range(y.shape[0]):
            # the first label in y as the first class while the other labels as the second class
            if y[i, 0] == 1:
                y_.append(1)
                pos_count += 1
            else:
                y_.append(0)
                neg_count += 1
        self.y = np.expand_dims(y_, axis=1)

    def find_class(self):
        dfs = []
        label_path = "NUS_WIDE/Groundtruth/TrainTestLabels/"
        for label in self.selected_labels_list:
            file = os.path.join(self.data_dir, label_path, "_".join(["Labels", label, self.data_type]) + ".txt")
            df = pd.read_csv(file, header=None)
            df.columns = [label]
            dfs.append(df)
        data_labels = pd.concat(dfs, axis=1)
        if len(self.selected_labels_list) > 1:
            selected = data_labels[data_labels.sum(axis=1) == 1]
        else:
            selected = data_labels
        features_path = "NUS_WIDE/NUS_WID_Low_Level_Features/Low_Level_Features"
        dfs = []
        for file in os.listdir(os.path.join(self.data_dir, features_path)):
            if file.startswith("_".join([self.data_type, "Normalized"])):
                df = pd.read_csv(os.path.join(self.data_dir, features_path, file), header=None, sep=" ")
                df.dropna(axis=1, inplace=True)
                print("{0} datasets features {1}".format(file, len(df.columns)))
                dfs.append(df)
        data_XA = pd.concat(dfs, axis=1)
        data_XA_selected = data_XA.loc[selected.index]
        print("XA shape:", data_XA_selected.shape)  # 634 columns

        # get XB, which are tags
        tag_path = "NUS_WIDE/NUS_WID_Tags/"
        file = "_".join([self.data_type, "Tags1k"]) + ".dat"
        tagsdf = pd.read_csv(os.path.join(self.data_dir, tag_path, file), header=None, sep="\t")
        tagsdf.dropna(axis=1, inplace=True)
        data_XB_selected = tagsdf.loc[selected.index]
        print("XB shape:", data_XB_selected.shape)
        return data_XA_selected.values, data_XB_selected.values, selected.values

    def __len__(self):
        return len(self.Xa)

    def __getitem__(self, index):  # this is single_indexx
        x_a = self.Xa[index]
        x_b = self.Xb[index]
        y = self.y[index]

        return [x_a, x_b], y


class MultiViewDataset6Party:
    def __init__(self, data_dir, data_type, height, width, k):
        self.x = []  # the datapath of k different png files
        self.y = []  # the corresponding label
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.k = k
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.89156885, 0.89156885, 0.89156885],
                                 std=[0.18063523, 0.18063523, 0.18063523]),
        ])

        self.classes, self.class_to_idx = self.find_class(data_dir)
        subfixes = ['_' + str(i).zfill(3) + '.png' for i in range(1, 13)]
        for label in self.classes:
            all_files = [d for d in os.listdir(os.path.join(data_dir, label, data_type))]
            all_off_files = [item.split('.')[0] for item in all_files if item[-3:] == 'off']
            all_off_files = sorted(list(set(all_off_files)))
            for single_off_file in all_off_files:
                all_views = [single_off_file + sg_subfix for sg_subfix in subfixes]
                all_views = [os.path.join(data_dir, label, data_type, item) for item in all_views]
                for i in range(2):
                    sample = [all_views[j + i * 6] for j in range(0, k)]
                    self.x.append(sample)
                    self.y.append([self.class_to_idx[label]])
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def find_class(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.x)

    def __getitem__(self, indexx):  # this is single_indexx
        _views = self.x[indexx]
        data = []
        labels = []
        for index in range(self.k):
            img = Image.open(_views[index])
            if self.transform is not None:
                img = self.transform(img)
            data.append(img)
        labels.append(self.y[indexx])

        return data, np.array(labels).ravel()


def test_dataset():
    DATA_DIR = './data'
    class_label_list = ['person', 'animal']
    train_dataset = NUS_WIDE_2_Party(DATA_DIR, class_label_list, 'Train', 2)
    # valid_dataset = NUS_WIDE_2_Party(DATA_DIR, class_label_list, 'Train', 2)
    n_train = len(train_dataset)
    # n_valid = len(valid_dataset)
    print(n_train)
    # print(n_valid)
    train_indices = list(range(n_train))
    # valid_indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    # valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=2,
                                               sampler=train_sampler,
                                               num_workers=0,
                                               pin_memory=False)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset,
    #                                            batch_size=2,
    #                                            sampler=valid_sampler,
    #                                            num_workers=2,
    #                                            pin_memory=True)
    print(len(train_loader))
    # print(len(valid_loader))
    for i, (x1, y) in enumerate(train_loader):
        print(y)
        print(x1[0].shape, y.shape)
        break


if __name__ == "__main__":
    test_dataset()
