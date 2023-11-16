import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from datetime import datetime
from re import I
import traceback
import time
import random
import torch
import torch.nn as nn
import numpy as np
import model
from utils import *
from model import *
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from torchvision.ops.focal_loss import sigmoid_focal_loss
from sklearn.metrics import recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from torch.autograd.profiler import profile, record_function, ProfilerActivity

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.utils import resample

import tqdm
from Datasets import *

def parse_args(cmd_dict):
    parser = argparse.ArgumentParser(description='train and valid')
    parser.add_argument('--config', default = 'default', type =str) # Read UniTS hyperparameters
    parser.add_argument('--dataset', default = 'widar3', type = str)
    parser.add_argument('--loader', default = 'dfs', type = str)
    parser.add_argument('--version', default = 'norm-filter', type = str)
    parser.add_argument('--model', default='THAT', type=str)
    # args.data_path_file default all.txt
    parser.add_argument('--data_path_file', default='all_top6.txt', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--log', default='log', type=str,
                        help="Log directory")
    parser.add_argument('--exp', default='', type=str)
    parser.add_argument('--ratio', default=0.2, type=float)
    parser.add_argument('--n_gpu', default=0, type =int)

    parser.add_argument('--test_ratio', default=0.5, type=float)
    
    parser.add_argument('--epochs', default = 200, type = int)
    parser.add_argument('--lr', default = 1e-3, type = float)
    parser.add_argument('--lr_decay_step', default = 1, type = int)
    parser.add_argument('--lr_decay_gamma', default = 1, type = float)
    parser.add_argument('--dry_run', action = 'store_true')

    parser.add_argument('--batch_size', default = 64, type = int)
    # dfs_win_size
    parser.add_argument('--dfs_win_size', default = 256, type = int)
    parser.add_argument('--es_patience', default = 30, type = int)

    parser.add_argument('--checkpoint', default = 'checkpoint.pt', type = str)
    # parser.add_argument('--save', action = 'store_true')
    parser.add_argument('--reload', action = 'store_true')
    parser.add_argument('--test_only', action = 'store_true')
    parser.add_argument('--enable_profiler', action = 'store_true')
    # args.es_enabled
    parser.add_argument('--enable_es', action = 'store_true')
    # enable tta
    parser.add_argument('--enable_test_aug', action = 'store_true')

    parser.add_argument('--time_aug', type=float, nargs="*", default=[])
    parser.add_argument('--freq_aug', type=str, nargs="*", default=[])
    parser.add_argument('--space_aug', type=str, nargs="*", default=[])

    parser.add_argument('--k_fold', default = 4, type = int)
    # max_fold
    parser.add_argument('--max_fold', default = 1, type = int)
    # use_cache
    parser.add_argument('--use_cache', action = 'store_true', default=True)
    parser.add_argument('--fake_data', action = 'store_true')
    args = parser.parse_args()
    
    machine_config = read_config("./config.yaml")
    update_args(args, machine_config)
    # run by another script
    update_args(args, cmd_dict)

    config = read_config(args.config + '.yaml')
    if not os.path.exists(args.log):
        os.mkdir(args.log)
    args.log_path = os.path.join(args.log, args.dataset)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    torch.cuda.set_device(args.n_gpu)

    if args.dataset == 'widar3-bvp':
        args.default_stft_window = args.dfs_win_size
        args.input_channel = 400
        args.input_size = 38
        args.batch_size = 64
        args.hheads = 9
        args.SENSOR_AXIS = 3
        args.num_labels = 6
        args.window_step = 10
        
        args.widar3_input_channel = 1
        args.widar3_input_x = 20
        args.widar3_input_y = 20
    elif args.dataset == 'widar3':
        args.default_stft_window = args.dfs_win_size
        args.batch_size = 128
        args.input_size = 256
        args.sample_rate = 1000

        args.hheads = 11 # input_channel % hheads == 0
        args.SENSOR_AXIS = 6
        args.num_labels = 6
        args.window_step = 10

        args.widar3_input_channel = 1
        args.widar3_input_x = 6
        args.widar3_input_y = 121

        args.input_channel = 726
        if args.exp.startswith("rx"):
            args.input_channel = 121
            args.SENSOR_AXIS = 1

        # (args.batch_size, args.input_size, args.widar3_input_channel, args.widar3_input_x, args.widar3_input_y)
        args.resnet2d_input_channel = 6
        if args.exp.startswith("rx"):
            args.resnet2d_input_channel = 1
        args.resnet2d_input_x = 128
        args.resnet2d_input_y = 128
        
        if args.model == 'CNN_GRU':
            args.resnet2d_input_channel = 1

    elif args.dataset == 'fall':
        args.default_stft_window = args.dfs_win_size
        args.window_step = 10
        args.sample_rate = 1000
        
        args.input_channel = 121
        args.input_size = 256

        args.batch_size = 64
        args.hheads = 11
        args.SENSOR_AXIS = 1
        args.num_labels = 2

        args.resnet2d_input_channel = 1
        args.resnet2d_input_x = 128
        args.resnet2d_input_y = 128
    elif args.dataset == 'widar3-csi':
        args.input_channel = 1080
        if args.exp.startswith("rx"):
            args.input_channel = 180
        args.input_size = 256
        args.batch_size = 128
        args.hheads = 9
        args.SENSOR_AXIS = 3
        args.num_labels = 6
        args.window_step = 10
        
        args.widar3_input_channel = 1
        args.widar3_input_x = 6
        args.widar3_input_y = 180

        args.resnet2d_input_channel = 6
        if args.exp.startswith("rx"):
            args.resnet2d_input_channel = 1
        args.resnet2d_input_x = 256
        args.resnet2d_input_y = 128
    return args, config

def my_worker_init_fn(worker_id):
    # random seed for each worker on each epoch
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_round_final(args):
    args.log("arg.time_aug: {}".format(args.time_aug))
    args.log("arg.freq_aug: {}".format(args.freq_aug))
    args.log("arg.space_aug: {}".format(args.space_aug))
    args.log("[Info]-> Model:{}, Dataset:{}, version:{}, arg.time_aug: {}, arg.freq_aug: {}, arg.space_aug: {}".format(args.model, args.dataset, args.version, args.time_aug, 
        args.freq_aug, args.space_aug))
    args.log("[Best_Valid_Acc, Best_Valid_F1] -> {:.7f}, {:.7f}".format(args.earlystoping.best_acc, args.earlystoping.best_f1))
    args.log("Best model stored @ " + str(args.earlystoping.best_path))
    args.log("End time:" + time.asctime(time.localtime(time.time())))

def test_round_final_log(args):
    pass
def choose_loss_function(args):
    return torch.nn.CrossEntropyLoss()

def get_model(args, config):
    if args.model == 'UniTS':
        model = UniTS(input_size = args.input_size, sensor_num = args.input_channel, layer_num = config.layer_num,
        window_list = config.window_list, stride_list = config.stride_list, k_list = config.k_list,
        out_dim = args.num_labels, hidden_channel = config.hidden_channel).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    elif args.model == 'static':
        model = static_UniTS(input_size = args.input_size, sensor_num = args.input_channel, layer_num = config.layer_num,
        window_list = config.window_list, stride_list = config.stride_list, k_list = config.k_list,
        out_dim = args.num_labels, hidden_channel = config.hidden_channel).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    elif args.model == 'THAT':
        args.hlayers = 5
        args.vlayers = 1
        args.vheads = 16
        args.K = 10
        args.sample = 4
        model = HARTrans(args).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)

    elif args.model == 'RFNet':
        model = RFNet(num_classes = args.num_labels, input_channel = args.input_channel, win_len = args.input_size).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)

    elif args.model == 'ResNet':
        model = ResNet(input_size = args.input_size, input_channel = args.input_channel, num_label = args.num_labels).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)

    elif args.model == 'MaDNN':
        args.lr_decay_step = 1
        args.lr_decay_gamma = 1
        model = MaDNN(input_size = args.input_size, input_channel = args.input_channel, num_label = args.num_labels).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)   

    elif args.model == 'MaCNN':
        args.lr_decay_step = 1
        args.lr_decay_gamma = 1
        model = MaCNN(input_size = args.input_size, input_channel = args.input_channel, num_label = args.num_labels, 
            sensor_num = int(args.input_channel / args.SENSOR_AXIS)).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)

    elif args.model == 'LaxCat':
        model = LaxCat(input_size = args.input_size, input_channel = args.input_channel, num_label = args.num_labels,
            hidden_dim = 64, kernel_size = 32, stride = 8).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)

    elif args.model == 'Widar3':
        # [@, T_MAX, 1, 20, 20]
        model = Widar3((args.batch_size, args.input_size, args.widar3_input_channel, args.widar3_input_x, args.widar3_input_y), input_channel = 1, num_label = args.num_labels, n_gru_hidden_units=128, f_dropout_ratio=0.5).cuda()
        # Use Adam beta1 = 0 and beta2 = 0.99 to reproduce the results in Widar3 paper.
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0, 0.99))

    elif args.model == 'ResNet18':
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(args.resnet2d_input_channel, 64, kernel_size = 3, stride = 1, padding = 1)
        model.fc = nn.Linear(512, args.num_labels)
        model.cuda()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)

    elif args.model == 'AlexNet':
        model = AlexNet(args.resnet2d_input_channel, num_classes = args.num_labels).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)

    elif args.model == 'CNN_GRU':
        model = CNN_GRU(args.resnet2d_input_channel, args.input_channel, args.num_labels).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    args.log('Total parameters: ' + str(total_params))

    return model, optimizer

def GetDataset(args):
    if args.loader == "dfs" or args.loader == "acf":
        dataset_class = RFBoostDataset
    elif args.loader == "csi":
        dataset_class = CSIDataset
    elif args.loader == "bvp":
        dataset_class = BVPDataset
    elif args.loader == "units":
        dataset_class = UnitsDataset
    else:
        raise ValueError('Unknown dataset: ' + args.dataset_name)
    return dataset_class

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def update_args(args, cmd_dict):
    if cmd_dict is not None:
        for key, value in cmd_dict.items():
            if value is not None and value != '':
                setattr(args, key, value)
def clear_cache():
    pass

def main(cmd_dict=None):
    
    args, config = parse_args(cmd_dict)
    
    log = set_up_logging(args, config)

    args.log = log
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")

    setup_seed(args.seed)
    start_time = time.time()
    log("Start time:" + time.asctime( time.localtime(time.time())) )
    log("arg.time_aug: {}".format(args.time_aug))
    log("arg.freq_aug: {}".format(args.freq_aug))
    log("arg.space_aug: {}".format(args.space_aug))

##############################################################################################################################
# make dataset
##############################################################################################################################
    train_kwargs = {'batch_size': args.batch_size, 
                    'shuffle': True, 
                    'num_workers': 16, 
                    # 'pin_memory': True, 
                    'drop_last': True,
                    'persistent_workers': False,
                    'prefetch_factor': 2,
                    'worker_init_fn': my_worker_init_fn,
                    }
                    
    eval_kwargs = { 'batch_size': args.batch_size,
                    'shuffle': False,
                    'num_workers': 16, 
                    # 'pin_memory': True, 
                    'drop_last': True,
                    'persistent_workers': False,
                    'prefetch_factor': 2, 
                    'worker_init_fn': my_worker_init_fn,
                    }
    # Use K-fold to shuffle training and validation set
    dataset_spliter = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    all_dataset = GetDataset(args)(args, os.path.join(args.dataset, args.data_path_file), 'all', config=config)
    log("all label: " + all_dataset.label_dist_str())
    
    # split train_valid and test, find their indices in all_dataset
    if args.test_ratio < 1:
        # all_dataset.apply_slice(list(range(512)))
        train_valid_index, test_index = train_test_split(list(range(len(all_dataset))), test_size=args.test_ratio, random_state=args.seed, shuffle=True, stratify=all_dataset.labels)
        # verbose
        # log("train_valid_index: " + str(train_valid_index))
        # log("test_index: " + str(test_index))  
    elif args.test_ratio == 1:
        train_valid_index, test_index = [], list(range(len(all_dataset)))

    # make train_valid and train datasets
    train_valid_dataset = GetDataset(args)(args, os.path.join(args.dataset, args.data_path_file), 'train_valid', slice_idx=train_valid_index, config=config)
    if args.enable_balance:
        # before balance
        log("Before balance: train_valid label: " + train_valid_dataset.label_dist_str())
        train_valid_index = train_valid_dataset.balance_label(train_valid_index)
        log("After balance: train_valid label: " + train_valid_dataset.label_dist_str())
    test_dataset = GetDataset(args)(args, os.path.join(args.dataset, args.data_path_file), 'test', slice_idx=test_index, config=config)
    test_loader = DataLoader(dataset=test_dataset, **eval_kwargs)

    if args.reload:
        model, optimizer = get_model(args, config)
        args.loss_func = choose_loss_function(args)
        # reload model
        if os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint))
        else:
            log("Model state dict not found!")
            return
        if args.test_only:
            eval_loss, eval_acc, eval_f1 = eval(model, test_loader, 0, "test", args)
            print("Test loss:{}, Test acc:{}, Test f1:{}".format(eval_loss, eval_acc, eval_f1))
            return
            
    # print("train_valid label: " + train_valid_dataset.label_dist_str())
    log("test label: " + test_dataset.label_dist_str())

    # split train and valid, find their indices in all_dataset
    for fold_i, (train_relative_index, valid_relative_index) in enumerate(dataset_spliter.split(train_valid_index, train_valid_dataset.labels)):
        if fold_i == args.max_fold:
            break
        train_index = [train_valid_index[i] for i in train_relative_index]
        valid_index = [train_valid_index[i] for i in valid_relative_index]
        
        # with open("split_info.txt", "a+") as f:
        #     f.write("Fold" + str(fold_i) + "\n")
        #     f.write(str(train_index))
        #     f.write("\n")
        #     f.write(str(valid_index))
        #     f.write("\n")
        #     f.write(str(test_index))
        #     f.write("\n")
        # continue
        # log("train_index: " + str(train_index))
        # log("valid_index: " + str(valid_index))

        fold_start_time = time.time()
        log('>>>>>>>> Fold {}/{} <<<<<<<<'.format(fold_i+1, args.k_fold))
        train_dataset = GetDataset(args)(args, os.path.join(args.dataset, args.data_path_file), "train", slice_idx=train_index, config=config)
        valid_dataset = GetDataset(args)(args, os.path.join(args.dataset, args.data_path_file), "valid", slice_idx=valid_index, config=config)
        
        log("train label: " + train_dataset.label_dist_str())
        log("valid label: " + valid_dataset.label_dist_str())

        train_loader = DataLoader(dataset=train_dataset, **train_kwargs)
        valid_loader = DataLoader(dataset=valid_dataset, **eval_kwargs)

        log("model: {}, dateset: {}".format(args.model, args.dataset))
        log('# of train samples: {}'.format(len(train_dataset)))
        log('# of valid samples: {}'.format(len(valid_dataset)))
        log('# of test samples: {}'.format(len(test_dataset)))
        
        # args.dry_run = True
        if args.dry_run:
            for x, y in tqdm.tqdm(train_loader):
                x = x.cuda(non_blocking=True).float()
                y = y.cuda(non_blocking=True).long()

            for x, y in tqdm.tqdm(valid_loader):
                x = x.cuda(non_blocking=True).float()
                y = y.cuda(non_blocking=True).long()

            for x, y in tqdm.tqdm(test_loader):
                x = x.cuda(non_blocking=True).float()
                y = y.cuda(non_blocking=True).long()
            continue

        ##############################################################################################################################
        # train & valid
        ##############################################################################################################################
        
        writer_loger_dir = "runs/{}/{}-{}-{}/TDA-{}_FDA-{}_SDA-{}_kfold-{}_{}" \
            .format(args.model, args.dataset, args.loader, args.version, args.time_aug, args.freq_aug, args.space_aug, fold_i+1, current_time)
        writer = SummaryWriter(log_dir = writer_loger_dir)
        args.writer = writer
        args.best_model_save_path = writer_loger_dir + "/best_model_{}.pt".format(fold_i+1)
        args.best_acc_save_path = writer_loger_dir + "/best_acc_{}.pt".format(fold_i+1)        
        args.earlystoping = EarlyStopping(patience=args.es_patience, verbose=True, best_path=args.best_model_save_path, best_acc_path=args.best_acc_save_path, es_enabled=args.enable_es)
        # make sure it is reproducible
        args.test_index_save_path = writer_loger_dir + "/test_index_{}.npy".format(fold_i+1)
        np.save(args.test_index_save_path, test_index)

        model, optimizer = get_model(args, config)
        if args.dataset == "fall":
            label_counter = Counter(train_valid_dataset.labels)
            label_ratio = np.array([label_counter[i] / len(train_valid_dataset.labels) for i in range(len(label_counter))])
            args.loss_func = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(1/label_ratio))
        else:
            args.loss_func = choose_loss_function(args)

        scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

        try:
            # train
            for ep in range(1, 1+args.epochs):
                epoch_start = time.time()
                # with record_function("train"):
                train(model, train_loader, optimizer, ep, args)
                # with record_function("eval"):
                eval_loss, eval_acc, eval_f1 = eval(model, valid_loader,  ep, "valid", args)
                args.earlystoping(eval_loss, eval_acc, eval_f1, ep, model)
                if args.earlystoping.early_stop:
                    log("Early stopping! Best valadLoss model(epoch={}) stored @ {}".format(args.earlystoping.best_ep_loss, str(args.earlystoping.best_path)))
                    log("Early stopping! Best valadAcc model(epoch={}) stored @ {}".format(args.earlystoping.best_ep_acc, str(args.earlystoping.best_acc_path)))
                    break
                log("This epoch used: " + str(time.time() - epoch_start))
                if type(scheduler) == ReduceLROnPlateau:
                    scheduler.step(eval_loss)
                else:
                    scheduler.step()
                clear_cache()
                log("----------------------------")
        except Exception as e:
            # stop
            traceback.print_exc()
            log('Exiting from training early')
            return
        finally:
            train_round_final(args)
            # reload from args.best_model_save_path
            if os.path.exists(args.best_model_save_path):
                model.load_state_dict(torch.load(args.best_model_save_path))
            else:
                log("Best Model state dict not found!")
                return
            # test
            _, test_acc, test_f1 = eval(model, test_loader, args.earlystoping.best_ep_loss, "test", args)
            log("Test on Best Validloss Model(epoch={})".format(args.earlystoping.best_ep_loss))
            log("Best Loss: [Test_Acc, Test_F1] -> {}, {}".format(test_acc, test_f1))

            if os.path.exists(args.best_acc_save_path):
                model.load_state_dict(torch.load(args.best_acc_save_path))
            else:
                log("Best Acc Model state dict not found!")
                return
            # test
            _, test_acc, test_f1 = eval(model, test_loader, args.earlystoping.best_ep_acc, "test", args)
            log("Test on Best Acc Model(epoch={})".format(args.earlystoping.best_ep_acc))
            log("Best Acc: [Test_Acc, Test_F1] -> {}, {}".format(test_acc, test_f1))
        
        log("Flod {} used: {}s".format(fold_i+1, time.time() - fold_start_time))
    # end k-fold loop
    log("End time:" + time.asctime(time.localtime(time.time())))
    log("Total time: " + str(time.time() - start_time))
    # end main

if __name__ == '__main__':
    main()
