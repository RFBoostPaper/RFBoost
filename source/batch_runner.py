import os
import subprocess
from datetime import datetime
import os, time
import subprocess
from async_timeout import timeout
import numpy as np
import argparse
from utils import compute_mean_and_conf_interval
from contextlib import redirect_stdout
from main import main
import sys, io
import traceback

class Recorder(object):
    def __init__(self, path, data_names=None, metric_names=None, method_names=None, model_names=None, description=None):
        self.start_time = datetime.now()
        self.path = path.replace('.txt', '_{}.txt'.format(self.start_time.strftime('%Y%m%d_%H%M%S')))

        self.data_names = data_names
        self.metric_names = metric_names
        self.method_names = method_names
        self.model_names = model_names    
        self.description = description
        
        self.result = [[[[-1.0 for _ in range(len(metric_names))] for _ in range(len(method_names))] for _ in range(len(data_names))] for _ in range(len(model_names))]
    
    def __enter__(self):        
        return self

    def record(self, data_key, metric_key, method_key, model_key, value):
        # find key in data_names
        data_idx = self.data_names.index(data_key)
        # find key in metric_names
        metric_idx = self.metric_names.index(metric_key)
        # find key in method_names
        method_idx = self.method_names.index(method_key)
        # find key in third_dim_names
        model_idx = self.model_names.index(model_key)
        # record value
        self.result[model_idx][data_idx][method_idx][metric_idx] = value


    def dump(self):
        with open(self.path, 'w+') as log_file:
            log_file.write(f'start from:{datetime.now()}\n')
            # write description
            log_file.write(f'description: {self.description}\n')
            # basic settings
            log_file.write(f'Adam betas=(0,0.99)\n')
            log_file.write(f'\n')
            for model_idx, model_name in enumerate(self.model_names):
                log_file.write(f'{model_name}:\n\n')
                # header            
                log_file.write("{:^20}|".format("Method"))
                for method in self.method_names:
                    log_file.write(f"{method:^63}|")
                log_file.write("\n")
                log_file.write("{:^20}|".format("Metrics"))
                for _ in range(len(self.method_names)):
                    for metric in self.metric_names:
                        log_file.write(f"{metric:^7}|")
                log_file.write("\n")
                log_file.write("\n")
                # body
                for data_idx, data_name in enumerate(self.data_names):
                    log_file.write(f'{data_name:<20}|')
                    for method_idx, method in enumerate(self.method_names):
                        for metric_idx, _ in enumerate(self.metric_names):
                            metric_value = self.result[model_idx][data_idx][method_idx][metric_idx]
                            # metric_value within [0, 1]
                            if metric_value > 0 and metric_value <= 1:
                                metric_value *= 100
                            log_file.write(f"{metric_value:^7.3f}|")
                    log_file.write('\n')
                log_file.write('\n')

            log_file.write(f'record dumped at:{datetime.now()}\n')
            log_file.write(f'elapsed:{datetime.now() - self.start_time}\n')

            log_file.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        self.dump()
        return False


if __name__ == "__main__":
    # exec bash "python main.py --dataset=widar3 --version=4 --es_patience=50 --model=ResNet"
    default_method_param = {"time_aug":"", "freq_aug":"", "space_aug":""}
    datasets = []
    models = []
    datasets += ["widar3"]
    models += ["ResNet","RFNet","ResNet18"]
    win_sizes = [256]

    versions = ["norm-filter"]
    # seeds = [0, 42, 2022, 1234, 5678]
    seeds = [0]
    # dataset x version
    datasets_versions = ["{}-{}".format(dataset, version) for dataset in datasets for version in versions]
    
    methods = {
                # "baseline":default_method_param,
                # "TDA-2": {**default_method_param, "time_aug":"0.5 2"},
                # "TDA-4": {**default_method_param, "time_aug":"0.5 0.75 1.5 2"},\
                # "TDA-6": {**default_method_param, "time_aug":"0.5 0.625 0.75 1.5 1.75 2"},\
                # "ms-top1": {**default_method_param, "freq_aug":"ms-top,1"},\
                # "ms-top2": {**default_method_param, "freq_aug":"ms-top,2"},\
                # "ms-top4": {**default_method_param, "freq_aug":"ms-top,4"},\
                # "ms-top6": {**default_method_param, "freq_aug":"ms-top,6"},\
                # "kmeans2": {**default_method_param, "freq_aug":"kmeans,2"},\
                # "kmeans4": {**default_method_param, "freq_aug":"kmeans,4"},\
                # "kmeans6": {**default_method_param, "freq_aug":"kmeans,6"},\
                # "ms-top1": {**default_method_param, "freq_aug":"ms-top,1"},\
                # "ms-top2": {**default_method_param, "freq_aug":"ms-top,2"},\
                # "ms-top4": {**default_method_param, "freq_aug":"ms-top,4"},\
                # "ms-top6": {**default_method_param, "freq_aug":"ms-top,6"},\
                # "ms-top8": {**default_method_param, "freq_aug":"ms-top,8"},\
                # "random-group,2": {**default_method_param, "freq_aug":"random-group,2"},\
                # "random-group,4": {**default_method_param, "freq_aug":"random-group,4"},\
                # "ms-even-div2": {**default_method_param, "freq_aug":"ms-even-div,2"},\
                # "ms-even-div4": {**default_method_param, "freq_aug":"ms-even-div,4"},
                # "ms-even-div6": {**default_method_param, "freq_aug":"ms-even-div,6"},   
                # "ms-even-div8": {**default_method_param, "freq_aug":"ms-even-div,8"},\
                # "subband-center1": {**default_method_param, "freq_aug":"subband-center,1"},\
                # "subband-center2": {**default_method_param, "freq_aug":"subband-center,2"},\
                # "subband-center3": {**default_method_param, "freq_aug":"subband-center,3"},\
                # "subband-center4": {**default_method_param, "freq_aug":"subband-center,4"},\
                # "subband-center6": {**default_method_param, "freq_aug":"subband-center,6"},\
                # "subband-ms-top1,3": {**default_method_param, "freq_aug":"subband-ms-top1,3"},
                # "kmeans-2+ms-even-div-2": {**default_method_param, "freq_aug":"kmeans,2 ms-even-div,2"},
                # "kmeans-2+ms-top-2": {**default_method_param, "freq_aug":"kmeans,2 ms-top,2"},
                # "TDA-2+kmeans-2+ms-top-2": {**default_method_param, "time_aug":"0.5 2", "freq_aug":"kmeans,2 ms-top,2"},
                # "TDA-2+kmeans-2+ms-even-div-2": {**default_method_param, "time_aug":"0.5 2", "freq_aug":"kmeans,2 ms-even-div,2"},
                # "kmeans-4+ms-top-4": {**default_method_param, "freq_aug":"kmeans,4 ms-top,4"},
                # "TDA-2+ms-top-2": {**default_method_param, "time_aug":"0.5 2", "freq_aug":"ms-top,2"},\
                "TDA-2+ms-top-4": {**default_method_param, "time_aug":"0.5 2", "freq_aug":"ms-top,4"},\
                # "TDA-4+kmeans-4": {**default_method_param, "time_aug":"0.5 0.75 1.5 2", "freq_aug":"kmeans,4"},\
                # "kmeans-2+ms-even-div-2": {**default_method_param, "time_aug":"0.5 2", "freq_aug":"kmeans,2 ms-even-div,2"},\
              }

    parser = argparse.ArgumentParser()
    # n_runs = 1
    parser.add_argument('--k_fold', type=int, default=4)
    # max_fold = k_fold
    parser.add_argument('--max_fold', type=int, default=4)
    # enable_es
    parser.add_argument('--enable_es', action = 'store_true', default=True)
    # exp
    parser.add_argument('--exp', type=str, default="")
    args = parser.parse_args()
    loader = "dfs"
    test_ratio = 0.5
    epochs = 100
    args.exp = ""
    args.enable_es = False
    enable_test_aug = False
    args.max_fold = 1
    enable_balance = False

    # descripbe purpose of the experiment
    exp_description = ""
    # test if /record exists
    if not os.path.exists('./record'):
        os.mkdir('./record')

    # begin to run experiments
    for seed in seeds:
        for dfs_win_size in win_sizes:
            with Recorder(path="record/results-Wsize-{}-seed-{}-max_fold-{}-ratio-{}-{}.txt".format(dfs_win_size, seed, args.max_fold, test_ratio, args.exp), 
                    description=exp_description,
                    data_names=datasets_versions, 
                    metric_names=["l_Acc", "l_Acc_e", "l_F1", "l_F1_e", "a_Acc", "a_Acc_e", "a_F1", "a_F1_e"],
                    method_names=list(methods.keys()),
                    model_names=models) as recorder:
                
                for model in models:
                    for dataset in datasets:
                        for version in versions:
                            for method_name, method_param in methods.items():
                                dataset_version = dataset + "-" + version
                                # clean cache
                                cmd =   "~/anaconda3/envs/rfboost/bin/python main.py"\
                                        " --dataset={dataset}"\
                                        " --version={version}"\
                                        " --k_fold={k_fold}"\
                                        " --model={model}"\
                                        " --time_aug {time_aug}"\
                                        " --freq_aug {freq_aug}"\
                                        " --space_aug {space_aug}"\
                                        " --seed={seed}"\
                                        " --dfs_win_size={dfs_win_size}"\
                                        " --max_fold={max_fold}"\
                                        " --exp={exp}"\
                                        " --epochs={epochs}"\
                                        .format(dataset=dataset, 
                                                version=version, 
                                                k_fold=args.k_fold,
                                                model=model,
                                                dfs_win_size=dfs_win_size,
                                                max_fold=args.max_fold,
                                                seed=seed,
                                                exp=args.exp,
                                                epochs=epochs,
                                                enable_test_aug=enable_test_aug,
                                                **method_param)
                                
                                cmd += " --enable_es" if args.enable_es else ""
                                cmd += " --enable_balance" if enable_balance else ""
                                cmd += " --enable_test_aug" if enable_test_aug else ""
                                # cmd += " --use_cache" if args.use_cache else ""
                                
                                print(cmd)
                                # convert variables to dict: dataset, version, k_fold, model, time_aug, freq_aug, space_aug, seed
                                cmd_dict = {
                                            "dataset":dataset,
                                            "version":version,
                                            "k_fold":args.k_fold,
                                            "model":model,
                                            "time_aug":'' if method_param["time_aug"] == '' else [float(m) for m in method_param["time_aug"].split(" ")],
                                            "freq_aug":'' if method_param["freq_aug"] == '' else method_param["freq_aug"].split(" "),
                                            "space_aug":method_param["space_aug"],
                                            "max_fold":args.max_fold,
                                            "dfs_win_size":dfs_win_size,
                                            "seed":seed,
                                            "epochs":epochs,
                                            "enable_es":args.enable_es,
                                            "enable_balance":enable_balance,
                                            "enable_test_aug":enable_test_aug,
                                            "exp":args.exp,
                                            "test_ratio":test_ratio,
                                            "loader": loader,
                                            }

                                # timer
                                start_time = time.time()
                                # try:
                                l_acc_list = []
                                l_f1_list = []
                                a_acc_list = []
                                a_f1_list = []
                                
                                output = io.StringIO()
                                # redirect stdout to string
                                with redirect_stdout(output):
                                    try:
                                        main(cmd_dict)
                                    except Exception as e:
                                        traceback.print_exc()
                                        print(output.getvalue())
                                        continue
                                # read stdout
                                output_lines = output.getvalue().splitlines()
                                for line in output_lines:
                                    if "Best Loss: [Test_Acc, Test_F1]" in line:
                                        l_acc = float(line.split("->")[-1].split(",")[0].strip())
                                        l_f1 = float(line.split("->")[-1].split(",")[1].strip())
                                        l_acc_list.append(l_acc)
                                        l_f1_list.append(l_f1)
                                    if "Best Acc: [Test_Acc, Test_F1]" in line:
                                        a_acc = float(line.split("->")[-1].split(",")[0].strip())
                                        a_f1 = float(line.split("->")[-1].split(",")[1].strip())
                                        a_acc_list.append(a_acc)
                                        a_f1_list.append(a_f1)
                                if len(l_acc_list) == 0:
                                    continue

                                print(l_acc_list, l_f1_list)
                                print(a_acc_list, a_f1_list)
                                l_acc, l_acc_err = compute_mean_and_conf_interval(l_acc_list)
                                l_f1, l_f1_err = compute_mean_and_conf_interval(l_f1_list)
                                recorder.record(dataset_version, "l_Acc", method_name, model, l_acc)
                                recorder.record(dataset_version, "l_Acc_e", method_name, model, l_acc_err)
                                recorder.record(dataset_version, "l_F1", method_name, model, l_f1)
                                recorder.record(dataset_version, "l_F1_e", method_name, model, l_f1_err)

                                a_acc, a_acc_err = compute_mean_and_conf_interval(a_acc_list)
                                a_f1, a_f1_err = compute_mean_and_conf_interval(a_f1_list)
                                recorder.record(dataset_version, "a_Acc", method_name, model, a_acc)
                                recorder.record(dataset_version, "a_Acc_e", method_name, model, a_acc_err)
                                recorder.record(dataset_version, "a_F1", method_name, model, a_f1)
                                recorder.record(dataset_version, "a_F1_e", method_name, model, a_f1_err)

                                recorder.dump()
                                print("--- %s seconds ---" % (time.time() - start_time))
