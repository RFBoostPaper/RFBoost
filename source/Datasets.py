from ast import arg
import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, average_precision_score
from sklearn.utils import resample
import scipy.io as sio
from utils import *
from augment import *
import queue as Queue
import threading
import torch.nn.functional as F
import tqdm
import mat73

from torchmetrics import AUROC
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader

from collections import Counter

class UnitsDataset(Dataset):
    def __init__(self, args, path_file, part, config=None, slice_idx=None):
        self.x, self.y = read_data(args, config, part)

        if args.dataset == "units-wifi":
            with open(path_file, 'r') as f:
                self.records = f.readlines()
                # if slice_idx is None, use ALL data,
                # otherwise, use slice_idx as index of part data
                slice_idx = list(range(len(self.records))) if slice_idx is None else slice_idx
                self.records = [self.records[i] for i in slice_idx]
                # make paths and labels
                self.data_paths = [record.split(' ')[0] for record in self.records]
                self.labels = [record.split(' ')[1].strip() for record in self.records]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

class BVPDataset(Dataset):
    def __init__(self, args, path_file, part, config=None, slice_idx=None):
        self.args = args
        self.aug_ratio = 3
        self.part = part
        with open(path_file, 'r') as f:
            self.records = f.readlines()

            # if slice_idx is None, use ALL data,
            # otherwise, use slice_idx as index of part data
            slice_idx = list(range(len(self.records))) if slice_idx is None else slice_idx
            self.records = [self.records[i] for i in slice_idx]
            self.data_paths = [record.split(' ')[0] for record in self.records]
            self.labels = [record.split(' ')[1].strip() for record in self.records]

    def label_dist_str(self):
        label_conter = Counter(self.labels)
        dist_str = "["
        # sort by label
        for label, cnt in sorted(label_conter.items(), key=lambda x: int(x[0])):
            dist_str += "{}:{:} ".format(label, cnt)
        dist_str += "]"
        return dist_str

    def apply_slice(self, slice_idx):
        if slice_idx is None:
            return 

        self.records = [self.records[i] for i in slice_idx]
        self.data_paths = [self.data_paths[i] for i in slice_idx]
        self.labels = [self.labels[i] for i in slice_idx]    

    def normalize_data(self, data_1):
        # data(ndarray)=>data_norm(ndarray): [20,20,T]=>[20,20,T]
        data_1_max = np.concatenate((data_1.max(axis=0),data_1.max(axis=1)),axis=0).max(axis=0)
        data_1_min = np.concatenate((data_1.min(axis=0),data_1.min(axis=1)),axis=0).min(axis=0)
        if (len(np.where((data_1_max - data_1_min) == 0)[0]) > 0):
            return data_1
        data_1_max_rep = np.tile(data_1_max,(data_1.shape[0],data_1.shape[1],1))
        data_1_min_rep = np.tile(data_1_min,(data_1.shape[0],data_1.shape[1],1))
        data_1_norm = (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep)
        return  data_1_norm

    def __getitem__(self, index):

        if self.part == 'train' or (self.part in ['test', 'valid'] and self.args.enable_test_aug):
            file_idx = index//self.aug_ratio
            aug_idx = index%self.aug_ratio
        else:
            file_idx = index

        data_path_relative = self.data_paths[file_idx]
        label = self.labels[file_idx]
        
        if self.part == 'train' or (self.part in ['test', 'valid'] and self.args.enable_test_aug):
            if aug_idx == 0:
                folder_name = "BVP-baseline-501"
                data_path_relative = data_path_relative.replace("Wsize251", "Wsize501")
            elif aug_idx == 1:
                folder_name = "BVP-125"
                data_path_relative = data_path_relative.replace("Wsize251", "Wsize125")
            elif aug_idx == 2:
                folder_name = "251"
        else:
            folder_name = "251"
        data_path = "../dataset/BVP/{}/{}.mat".format(folder_name, data_path_relative)
        # label start from 0
        label = int(label) - 1

        # [X, Y, T]
        try:
            bvp = sio.loadmat(data_path)['velocity_spectrum_ro']
        except Exception as e:
            self.args.log(e)
            self.args.log("Error: {}".format(data_path))
            raise Exception("Error: {}".format(data_path))


        bvp = self.normalize_data(bvp)
        # [X, Y, T] -> [T, X, Y]
        bvp = np.transpose(bvp, (2, 0, 1)) 
        # do padding only for time dimension
        # [T, X, Y] -> [input_size, X, Y]
        if self.args.input_size > bvp.shape[0]:
            bvp = np.pad(bvp, ((0, self.args.input_size - bvp.shape[0]), (0, 0), (0, 0)), 'constant')
        else:
            bvp = bvp[:self.args.input_size, :, :]

        # [T, X, Y] -> [T, 1, X, Y]
        bvp = np.expand_dims(bvp, axis=1)
        return bvp, label

    def __len__(self):
        if self.part == 'train' or (self.part in ['test', 'valid'] and self.args.enable_test_aug):
            return len(self.data_paths) * self.aug_ratio
        else:
            return len(self.records)

class CSIDataset(Dataset):
    def __init__(self, args, path_file, part, config=None, slice_idx=None):
        self.args = args
        if self.args.dataset == "signfi":
            self.augment = Augmentation(args.default_stft_window)
            try:
                self.records = sio.loadmat(os.path.join("../dataset/signfi/", 'dataset_lab_276_dl_{}.mat').format(self.args.version))
            except:
                self.records = mat73.loadmat(os.path.join("../dataset/signfi/", 'dataset_lab_276_dl_{}.mat').format(self.args.version))
            try:
                self.labels = sio.loadmat(os.path.join("../dataset/signfi/", 'dataset_lab_276_dl_label_fix.mat'))['label']
            except: 
                self.labels = mat73.loadmat(os.path.join("../dataset/signfi/", 'dataset_lab_276_dl_label_fix.mat'))['label']
            
            self.ms = self.records['ms']
            self.records = self.records['data_pp']

            slice_idx = list(range(len(self.records))) if slice_idx is None else slice_idx
            self.records = [self.records[i] for i in slice_idx]
            self.labels = [int(self.labels[i]) for i in slice_idx]
            self.ms = [self.ms[i] for i in slice_idx]
        else:
            with open(path_file, 'r') as f:
                self.records = f.readlines()

                # if slice_idx is None, use ALL data,
                # otherwise, use slice_idx as index of part data
                slice_idx = list(range(len(self.records))) if slice_idx is None else slice_idx
                self.records = [self.records[i] for i in slice_idx]

                # make paths and labels
                self.data_paths = [record.split(' ')[0] for record in self.records]
                self.labels = [record.split(' ')[1].strip() for record in self.records]
    
    def apply_slice(self, slice_idx):
        if slice_idx is None:
            return 

        self.records = [self.records[i] for i in slice_idx]
        self.data_paths = [self.data_paths[i] for i in slice_idx]
        self.labels = [self.labels[i] for i in slice_idx]
    
    def label_dist_str(self):
        label_conter = Counter(self.labels)
        dist_str = "["
        # sort by label
        for label, cnt in sorted(label_conter.items(), key=lambda x: int(x[0])):
            dist_str += "{}:{:} ".format(label, cnt)
        dist_str += "]"
        return dist_str

    def __getitem__(self, index):
        if self.args.dataset in ["signfi"]:
            # T, F
            csi_data, ms = self.records[index], self.ms[index]
            # T, F -> T, F, 1
            csi_data = np.expand_dims(csi_data, axis=2)
            label = self.labels[index]
            if self.args.dataset in ["signfi"]:
                label = int(label) - 1
        else:
            # data_path, label = self.records[index].split(' ')
            data_path = self.data_paths[index]
            label = self.labels[index]
            # data_path = "~/Documents/RFBoost/dataset/NPZ-"+ self.args.version + "/" + data_path + ".npz"
            data_path = "../dataset/NPZ-pp/{}/{}.npz".format(self.args.version, data_path)
            # label start from 0
            label = int(label) - 1
        
            # [T, F, Rx]
            try:
                csi_data, _ = read_npz_data(data_path)
            except Exception as e:
                print(e)
                self.args.log("Error: {}".format(data_path))
                raise Exception("Error: {}".format(data_path))

        # csi_data = sio.loadmat(data_path)['data']
        # csi_data *= 1e6
        if self.args.exp.startswith("rx"):
            rx_sel = int(self.args.exp.split('-')[1])
            csi_data = csi_data[:, :, rx_sel][:, :, np.newaxis]
        csi_data_amp = np.abs(csi_data)
        csi_data_phase = np.angle(csi_data)
        # standardization respectively
        csi_data_amp = (csi_data_amp - np.mean(csi_data_amp)) / np.std(csi_data_amp)
        csi_data_phase = (csi_data_phase - np.mean(csi_data_phase)) / np.std(csi_data_phase)
        csi_data = np.concatenate((csi_data_amp, csi_data_phase), axis=1)

        if csi_data.shape[0] < self.args.input_size:
            csi_data = np.pad(csi_data, ((0, self.args.input_size - csi_data.shape[1]), (0, 0), (0,0)), 'constant')
        
        # [T, F, Rx] -> [Rx, T, F]
        csi_data = csi_data.transpose((2, 0, 1))
        csi_data = np.array([pad_and_downsample(d, self.args.input_size, axis=0) for d in csi_data])

        if self.args.model == "Widar3":
            # [Rx, T, F] -> [T, Rx, F]
            csi_data = csi_data.transpose((1, 0, 2))
            # [T, Rx, F] -> [T, 1, Rx, F]
            csi_data = np.expand_dims(csi_data, axis=1)
        elif self.args.model in ["ResNet18", "AlexNet"]:
            pass
            # csi_data = np.array([pad_and_downsample(d, self.args.resnet2d_input_x, axis=1) for d in csi_data])
        else:
            # [Rx, T, F] -> [T, Rx, F]
            csi_data = csi_data.transpose((1, 0, 2))
            # [T, Rx, F] -> [T, Rx * F]
            csi_data = csi_data.reshape((csi_data.shape[0], -1))

        if self.args.model == "CNN_GRU":
            # [W, Rx*F] -> [W, 1, Rx*F]
            csi_data = np.expand_dims(csi_data, axis=1)

        return csi_data, label

    def __len__(self):
        return len(self.records)

class RFBoostDataset(Dataset):
    def __init__(self, args, path_file, part, config=None, slice_idx=None):
        self.part = part
        self.args = args
        self.time_aug = args.time_aug
        self.freq_aug_list = []
        self.freq_args_list = []
        
        if args.freq_aug != [] and args.freq_aug != ['']:
            # e.g. ["kmeans,4", "ms-top,4"]
            for freq_aug_one in args.freq_aug:
                # "kmeans,4"
                fda_policy, freq_args = freq_aug_one.split(',')
                freq_args = int(freq_args)
                for i in range(freq_args):
                    self.freq_aug_list.append(fda_policy)
                    self.freq_args_list.append((i, freq_args))
        else:
            self.freq_args_list = []

        # Augmentation parameters
        self.space_aug = args.space_aug

        self.n_fda = len(self.freq_aug_list)
        self.n_tda = len(self.time_aug)
        self.n_sda = len(self.space_aug)

        self.augment = Augmentation(args.default_stft_window, window_step=10)
        self.aug_ratio = 1 + self.n_fda + self.n_tda + self.n_sda
        if self.args.dataset == "fall":
            normal_data, normal_ms = read_any_data(os.path.join("../dataset/fall/", 'normal_3312.npz'))
            fall_data, fall_ms = read_any_data(os.path.join("../dataset/fall/", 'fall_442.npz'))

            label_0 = np.zeros(len(normal_data))
            label_1 = np.ones(len(fall_data))

            self.records = np.concatenate((normal_data, fall_data), axis=0)
            self.labels = np.concatenate((label_0, label_1), axis=0)
            self.ms = np.concatenate((normal_ms, fall_ms), axis=0)

            # if slice_idx is None, use ALL data,
            # otherwise, use slice_idx as index of part data
            slice_idx = list(range(len(self.records))) if slice_idx is None else slice_idx
            self.records = np.array([self.records[i] for i in slice_idx])
            self.labels = np.array([int(self.labels[i]) for i in slice_idx])
            self.ms = np.array([self.ms[i] for i in slice_idx])
        elif self.args.dataset == "signfi":
            self.augment = Augmentation(args.default_stft_window)
            try:
                self.records = sio.loadmat(os.path.join("../dataset/signfi/", 'dataset_lab_276_dl_norm.mat'))
            except:
                self.records = mat73.loadmat(os.path.join("../dataset/signfi/", 'dataset_lab_276_dl_norm.mat'))
            try:
                self.labels = sio.loadmat(os.path.join("../dataset/signfi/", 'dataset_lab_276_dl_label_fix.mat'))['label']
            except: 
                self.labels = mat73.loadmat(os.path.join("../dataset/signfi/", 'dataset_lab_276_dl_label_fix.mat'))['label']
            
            self.ms = self.records['ms']
            self.records = self.records['data_pp']

            slice_idx = list(range(len(self.records))) if slice_idx is None else slice_idx
            self.records = [self.records[i] for i in slice_idx]
            self.labels = [int(self.labels[i]) for i in slice_idx]
            self.ms = [self.ms[i] for i in slice_idx]
        else:
            # Load path file
            '''
            usage:
            all_dataset = RFBoostDataset(args, path_file, "all")
            for i, (train_index, valid_index) = enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=0)):
                train_dataset = RFBoostDataset(args, path_file, "train", train_index)
                valid_dataset = RFBoostDataset(args, path_file, "test", valid_index)
            '''
            with open(path_file, 'r') as f:
                self.records = f.readlines()
                # if slice_idx is None, use ALL data,
                # otherwise, use slice_idx as index of part data
                slice_idx = list(range(len(self.records))) if slice_idx is None else slice_idx
                self.records = [self.records[i] for i in slice_idx]
                # make paths and labels
                self.data_paths = [record.split(' ')[0] for record in self.records]
                self.labels = np.array([int(record.split(' ')[1].strip())-1 for record in self.records])

    
    def apply_slice(self, slice_idx):
        if slice_idx is None:
            return 

        self.records = [self.records[i] for i in slice_idx]
        self.data_paths = [self.data_paths[i] for i in slice_idx]
        self.labels = [self.labels[i] for i in slice_idx]
        self.ms = [self.ms[i] for i in slice_idx]
    
    def balance_label(self, absolute_idx_to_all):
        if self.part not in ["train", "valid", "train_valid"]:
            return
        # balance label: 
        # only test on FallDar
        # group by label
        
        all_idx = np.array(list(range(len(self.labels))))
        idx_group_by_label = {label:all_idx[self.labels==label] for label in range(self.args.num_labels)}
        cnt_group_by_label = {label:len(idx_group_by_label[label]) for label in range(self.args.num_labels)}
        # counter = Counter(self.labels)
        max_count = max(cnt_group_by_label.values())
        for label in range(self.args.num_labels):
            # oversample
            num_add = max_count - cnt_group_by_label[label]
            if num_add > 0:
                np.random.seed(self.args.seed)
                idx_add = np.random.choice(idx_group_by_label[label], size=num_add, replace=True)
                absolute_idx_to_all.extend(idx_add)
                self.records = np.concatenate([self.records, self.records[idx_add]], axis=0)
                self.labels = np.concatenate([self.labels, self.labels[idx_add]], axis=0)
                self.ms = np.concatenate([self.ms, self.ms[idx_add]], axis=0)
                cnt_group_by_label[label] = max_count

        return absolute_idx_to_all

        
    def label_dist_str(self):
        label_conter = Counter(self.labels)
        dist_str = "["
        # sort by label
        for label, cnt in sorted(label_conter.items(), key=lambda x: int(x[0])):
            dist_str += "{}:{:} ".format(label, cnt)
        dist_str += "]"
        return dist_str
    

    def __getitem__(self, index):
        """
        index mapping to the original data
        len(time_aug) + len(freq_aug) + len(space_aug) = aug_ratio - 1
        0...         a1-1 ..      a1+a2-1...        a1+a2+a3-1... a1+a2+a3(original)

        Augmented data index:
        0   1   2   3   ...   aug_ratio-1
        aug             ...     
        2*aug

        ...
        (n-1)*aug     ...     n*aug_ratio-1
        
        """
        # if augmentation is enabled
        if self.args.exp=="imb":
            # don't use augmentation for all
            file_idx = index
            # original data
            aug_idx = self.aug_ratio 
        else:
            if self.part == 'train' or (self.part in ['test', 'valid'] and self.args.enable_test_aug):
                file_idx = index//self.aug_ratio
                aug_idx = index%self.aug_ratio
            else:
                file_idx = index
        
        # Get Spectrogram
        try:
            if self.args.dataset in ["signfi", "fall"]:
                # T, F
                csi_data, ms = self.records[file_idx], self.ms[file_idx]
                # T, F -> T, F, 1
                csi_data = np.expand_dims(csi_data, axis=2)
                label = self.labels[file_idx]
                if self.args.dataset in ["signfi"]:
                    label = int(label) - 1
            elif self.args.dataset.startswith("widar"):
                data_path = self.data_paths[file_idx]
                label = self.labels[file_idx]

                data_path = "../dataset/NPZ-pp/{}/{}.npz".format(self.args.version, data_path)
                
                # [T, F, Rx]
                try:
                    csi_data, ms = read_npz_data(data_path)
                except Exception as e:
                    self.args.log(e)
                    self.args.log("Error: {}".format(data_path))
                    raise Exception("Error: {}".format(data_path))

                if self.args.exp.startswith("rx"):
                    rx_sel = int(self.args.exp.split('-')[1])
                    csi_data = csi_data[:, :, rx_sel][:, :, np.newaxis]
                    ms = ms[:, rx_sel][:, np.newaxis]
            # Data padding
            if self.args.dataset in ["signfi", "fall"]:
                pass
            elif csi_data.shape[0] < 512:
                csi_data = np.pad(csi_data, ((0, 512 - csi_data.shape[0]), (0, 0), (0,0)), 'constant')
            elif csi_data.shape[0] > 5120:
                ds_rate = 10
                csi_data = csi_data[:, ::ds_rate, :]   

            # with record_function("transpose Rx"):
            # [T, F, Rx] -> [Rx, T, F]
            csi_data = np.transpose(csi_data, (2, 0, 1))
            if ms is not None:
            # if ms has only 1 dim
                if ms.ndim == 1:
                    ms = np.expand_dims(ms, axis=0)
                else:
                    # # [F, Rx] -> [Rx, F]
                    ms = np.transpose(ms, (1, 0))

            # with record_function("Get_DFS"):
            # Augmentation
            if self.part == 'train' or (self.part in ['test', 'valid'] and self.args.enable_test_aug):
            # if self.args.enable_test_aug and not self.part == 'all':
                if self.args.exp=="imb" and label == 0:
                    # index for label 0
                    idx_label_0 = np.where(self.labels==0)[0]
                    # downsample to 1/10
                    # idx_label_0_ds = idx_label_0_ds[::self.aug_ratio]
                    label_0_this_intra_order = np.where(idx_label_0==file_idx)[0][0]

                    file_idx = idx_label_0[(label_0_this_intra_order//self.aug_ratio)*self.aug_ratio]
                    # aug_idx = label_0_this_intra_order%self.aug_ratio
                    # running:
                    aug_idx = self.aug_ratio
                if aug_idx < self.n_tda:
                    tda_idx = aug_idx
                    dfs = [self.augment.time_augment(csi_data[i], ms[i], self.time_aug[tda_idx], args=self.args, agg_type="pca")[2] for i in range(csi_data.shape[0])] 
                elif aug_idx < self.n_tda + self.n_fda:
                    fda_idx = aug_idx - self.n_tda 
                    dfs = [self.augment.frequency_augment(csi_data[i], ms[i], self.freq_aug_list[fda_idx], self.freq_args_list[fda_idx][1], fda_th=self.freq_args_list[fda_idx][0], file_th=file_idx, rx_th=i, args=self.args)[2] for i in range(csi_data.shape[0])]
                elif aug_idx < self.n_tda + self.n_fda + self.n_sda:
                    # use frequency augmentation only.
                    sda_idx = aug_idx - self.n_tda - self.n_fda
                    dfs = [self.augment.space_augment(csi_data[i], ms[i], self.space_aug[sda_idx], args=self.args)[2] for i in range(csi_data.shape[0])]
                else:
                    dfs = [get_dfs(csi_data[i], ms[i], samp_rate=self.args.sample_rate, window_size=self.args.default_stft_window, window_step=self.args.window_step, agg_type="pca", cache_folder=self.args.cache_folder)[2] for i in range(csi_data.shape[0])]
            else:
                # test (when disable TTA)
                dfs = [get_dfs(csi_data[i], ms[i], samp_rate=self.args.sample_rate, window_size=self.args.default_stft_window, window_step=self.args.window_step, agg_type="pca", cache_folder=self.args.cache_folder)[2] for i in range(csi_data.shape[0])]

            # with record_function("pad_ds"):
            # if dim_num == 3
            if np.array(dfs).shape.__len__() != 3:
                pass
            
            # [Rx, F, W]
            dfs = np.array([pad_and_downsample(d, self.args.input_size) for d in dfs])
            
            if self.args.model == 'Widar3':
                # [Rx, F, W] -> [W, Rx, F] 
                dfs = dfs.transpose((2, 0, 1))
                # [W, Rx, F] -> [W, 1, Rx, F] extend_dim
                dfs = np.expand_dims(dfs, axis=1)
            elif self.args.model in ["ResNet18", "AlexNet"]:
                if self.args.exp.startswith("rx"):
                    dfs = np.array([pad_and_downsample(d, self.args.resnet2d_input_x, axis=0) for d in dfs])
                # [Rx, F, W] -> [Rx, W, F]
                dfs = dfs.transpose((0, 2, 1))
                # [Rx, W, F] -> [W, Rx, F]
                if self.args.model == "CNN_GRU":
                    dfs = dfs.transpose((1, 0, 2))
                    # [Rx, W, F] -> [W, F, Rx]
                    # dfs = dfs.transpose((1, 2, 0))
            else:
                # [Rx, F, W] -> [W, Rx, F] 
                dfs = dfs.transpose((2, 0, 1))
                # [W, Rx, F] -> [W, Rx*F] 
                dfs = dfs.reshape((dfs.shape[0], -1))
                # abs
                # dfs = np.abs(dfs)
                if self.args.model == "CNN_GRU":
                    # [W, Rx*F] -> [W, 1, Rx*F]
                    dfs = np.expand_dims(dfs, axis=1)
            
            # normalize
            dfs = (dfs - np.mean(dfs)) / np.std(dfs)

            if np.isnan(dfs).any():
                print('nan')
                # print(data_path)
            return dfs, label
        except Exception as e:
            self.args.log(e)
            self.args.log("Error: {}".format(data_path))
            raise Exception("Error: {}".format(data_path))

    def __len__(self):
        if self.args.exp=="imb":
            return len(self.records)

        if self.part == 'train' or (self.part in ['test', 'valid'] and self.args.enable_test_aug):
        # 
        # if self.args.enable_test_aug and not self.part == 'all':
            return len(self.records) * self.aug_ratio
        else:
            return len(self.records)

def eval(model, eval_loader, epoch, kind, args):
    y_pred = []
    y_true = []
    prob_all =[]
    eval_loss = 0
    iter_conter = 0
    with torch.no_grad():
        model.eval()
        for i, (x, y) in enumerate(tqdm.tqdm(eval_loader)):
            iter_conter += 1
            x = x.cuda(non_blocking=True).float()
            y = y.cuda(non_blocking=True).long()

            out = model(x)
            prob = F.softmax(out, dim=1)
            prob_all.append(prob.cpu().numpy())
            # eval_loss += args.loss_func(out, y)
            loss = args.loss_func(out, y)
            eval_loss += loss.cpu().item()
            
            pred = torch.argmax(out, dim = -1)
            y_pred += pred.cpu().tolist()
            y_true += y.cpu().tolist()

        eval_loss /= iter_conter
    
    if args.num_labels == 2:
        # calculate ROC curve using out
        prob_all = np.concatenate(prob_all, axis=0)
        fpr, tpr, thresholds = roc_curve(y_true, prob_all[:, 1])
        args.log("ROC fpr: {}, tpr: {}, thresholds: {}".format(fpr, tpr, thresholds))
        auc = roc_auc_score(y_true, prob_all[:, 1], average='weighted')
        # fpr80
        fpr80 = fpr[np.where(tpr >= 0.8)[0][0]]
        fpr95 = fpr[np.where(tpr >= 0.95)[0][0]]
        fnr = 1 - tpr
        eer_threshold = thresholds[np.argmin(np.absolute(fnr - fpr))]
        eer_pred = prob_all[:, 1] >= eer_threshold
        y_pred = eer_pred.astype(int)
        auprc = average_precision_score(y_true, prob_all[:, 1], average='weighted')
        
        



    eval_acc = accuracy_score(y_true, y_pred)
    if args.num_labels == 2:
        eval_f1 = f1_score(y_true, y_pred, average='weighted')
    else:
        eval_f1 = f1_score(y_true, y_pred, labels=list(range(args.num_labels)),average='macro')
    """
    draw a confusion matrix with TF, TN, FP, FN:
    GT\Pred |Positive | Negative
    ------------------+-----------
    True    | TP      | FN          # <- Fall
    False   | FP      | TN          # <- Normal
    
    false alarm rate = FP / (FP + TN)
    miss alarm rate = FN / (FN + TP)
    detection rate = TP / (TP + FN)

    FPR = FP / (FP + TN)
    TPR = TP / (TP + FN)

    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    # false alarm rate: number of misidentified normal instances and the number of all normal instances
    false_alarm_rate = np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0)
    # miss alarm rate: the number of misidentified fall instances and the number of all fall instances
    miss_alarm_rate = np.sum((y_true == 1) & (y_pred == 0)) / np.sum(y_true == 1)
    detection_rate = 1 - miss_alarm_rate
    if kind == "valid":
        args.log("[epoch={}]Validation Accuracy : {:.7} Macro F1 : {:.7} Loss : {:.7}\n".
            format(epoch, str(eval_acc), str(eval_f1), str(eval_loss)))
        args.writer.add_scalar('accuracy/valid', eval_acc, epoch)
        args.writer.add_scalar('f1/valid', eval_f1, epoch)
        args.writer.add_scalar('loss/valid', eval_loss, epoch)
        args.writer.add_scalar('FAR/valid', false_alarm_rate, epoch)
        args.writer.add_scalar('MAR/valid', miss_alarm_rate, epoch)
        if args.num_labels == 2:
            args.writer.add_scalar('AUC/valid', auc, epoch)
            args.writer.add_scalar('AUPRC/valid', auprc, epoch)
            # detection_rate
            args.writer.add_scalar('DET/valid', detection_rate, epoch)
            # args.writer.add_scalar('EER/valid', eer, epoch)
            args.writer.add_scalar('FPR80/valid', fpr80, epoch)
            args.writer.add_scalar('FPR95/valid', fpr95, epoch)

    elif kind == "test":
        args.log("[epoch={}]Test Accuracy : {:.7} Macro F1 : {:.7} Loss : {:.7}\n".
            format(epoch, str(eval_acc), str(eval_f1), str(eval_loss)))
        args.writer.add_scalar('accuracy/test', eval_acc, epoch)
        args.writer.add_scalar('f1/test', eval_f1, epoch)
        args.writer.add_scalar('loss/test', eval_loss, epoch)
        args.writer.add_scalar('FAR/test', false_alarm_rate, epoch)
        args.writer.add_scalar('MAR/test', miss_alarm_rate, epoch)
        if args.num_labels == 2:
            args.writer.add_scalar('AUC/test', auc, epoch)
            args.writer.add_scalar('AUPRC/test', auprc, epoch)
            # args.writer.add_scalar('EER/valid', eer, epoch)
            args.writer.add_scalar('DET/test', detection_rate, epoch)
            args.writer.add_scalar('FPR80/test', fpr80, epoch)
            args.writer.add_scalar('FPR95/test', fpr95, epoch)

    # confusion matrix
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(args.num_labels)))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    return eval_loss, eval_acc, eval_f1

def train(model, train_loader, optimizer, epoch, args):
    y_pred = []
    y_true = []
    epoch_loss = 0
    iter_conter = 0
    x_counter = 0
    for i, (x, y) in enumerate(tqdm.tqdm(train_loader)):
        iter_conter += 1
        model.train()

        x = x.cuda(non_blocking=True).float()
        y = y.cuda(non_blocking=True).long()

        out = model(x)
        loss = args.loss_func(out, y)
        if np.isnan(loss.item()):
            print("Loss is NaN")
            exit()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        epoch_loss += loss.cpu().item()

        pred = torch.argmax(out, dim = -1)
        y_pred += pred.cpu().tolist()
        y_true += y.cpu().tolist()

        x_counter += len(x)
        if (i != 0 and (i+1) % (len(train_loader.dataset)//4*args.batch_size) == 0) or x_counter == (len(train_loader.dataset)-1):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}'.format(
                epoch, x_counter, len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))

    epoch_loss /= iter_conter
    train_acc = accuracy_score(y_true, y_pred)
    train_f1 = f1_score(y_true, y_pred, labels=list(range(args.num_labels)),average='macro')

    args.writer.add_scalar('loss/train', epoch_loss, epoch)
    args.writer.add_scalar('accuracy/train', train_acc, epoch)
    args.writer.add_scalar('f1/train', train_f1, epoch)

    args.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    
    args.log("End of Epoch : " + str(epoch) + " Loss(avg) : " + str(epoch_loss))
