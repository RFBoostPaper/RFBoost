from ast import AugStore
from cProfile import label
from select import select
import numpy as np
from sklearn.cluster import KMeans
from utils import *
import random
import sys, time
class Augmentation(object):
    """
    Augmentation: output augmented spectrograms using registered RF data

    usage:
        # prepare processed RF data
        augmentation = Augmentation()

    """
    
    def __init__(self, default_stft_window=256, window_step=10):
        
        self.default_stft_window = default_stft_window
        self.kmeans_grp = dict()
        self.random_grp = dict()

    def time_augment(self, data, ms, ratio, agg_type="ms", args = None):
        window_size = int(self.default_stft_window * ratio)
        # print("aug window: ", window_size)
        f,t,augmented = get_dfs(data, ms, samp_rate=args.sample_rate, window_size=window_size, agg_type=agg_type, window_step=args.window_step, cache_folder=args.cache_folder)
        return f,t,augmented


    def frequency_augment(self, data, ms, method, param = None, args=None, fda_th=None, file_th=None, rx_th=None):
        """
        data: (ts, sub_num)
        ms: (sub_num
        
        """
        data = np.array(data)
        ms = np.array(ms)

        f_list = []
        t_list = []
        spectrograms = []
        augmented = []
        ms_list = []

        if method.startswith("ida"):
            method = method.replace("ida_", "")

            f,t,spec = get_dfs(data, ms, samp_rate=args.sample_rate, window_size=args.default_stft_window, agg_type="pca", window_step=args.window_step, cache_folder=args.cache_folder)
            

            if method == "time-shift":
                # [F, T]
                time_len = spec.shape[1]
                shift_len = time_len // param
                # 1:param: time_len
                shift_list = np.arange(0, time_len, shift_len)

                this_shift = shift_list[fda_th]
                augmented = self.shift_augment(spec, 0, this_shift)

                return f,t,augmented
            elif method == "freq-shift":
                # [F, T]
                freq_len = spec.shape[0]
                shift_len = freq_len // param
                # 1:param: freq_len
                shift_list = np.arange(0, freq_len, shift_len)

                this_shift = shift_list[fda_th]
                augmented = self.shift_augment(spec, this_shift, 0)

                return f,t,augmented
            elif method == "time-flip":
                augmented = np.flip(spec, axis=1)
                return f,t,augmented

            elif method == "freq-flip":
                augmented = np.flip(spec, axis=0)
                return f,t,augmented

            elif method == "flip-both":
                augmented = np.flip(spec, axis=(0,1))
                return f,t,augmented

            elif method == "time-flip-shift":
                spec = np.flip(spec, axis=1)
                # [F, T]
                time_len = spec.shape[1]
                shift_len = time_len // param
                # 1:param: time_len
                shift_list = np.arange(0, time_len, shift_len)

                this_shift = shift_list[fda_th]
                augmented = self.shift_augment(spec, 0, this_shift)
                return f,t,augmented

            elif method == "freq-flip-shift":
                spec = np.flip(spec, axis=0)
                # [F, T]
                freq_len = spec.shape[0]
                shift_len = freq_len // param
                # 1:param: freq_len
                shift_list = np.arange(0, freq_len, shift_len)

                this_shift = shift_list[fda_th]
                augmented = self.shift_augment(spec, this_shift, 0)
                return f,t,augmented

            elif method == "mask-time":
                # randomly zero out some time points
                # [F, T]
                time_len = spec.shape[1]
                mask_len = time_len // param
                # 1:param: time_len
                mask_list = np.arange(0, time_len, mask_len)

                this_mask = mask_list[fda_th]
                spec[:, this_mask:this_mask + mask_len] = 0
                return f,t,spec
                
            
            elif method == "mask-freq":
                # randomly zero out some frequency points
                # [F, T]
                freq_len = spec.shape[0]
                mask_len = freq_len // param
                # 1:param: freq_len
                mask_list = np.arange(0, freq_len, mask_len)

                this_mask = mask_list[fda_th]
                spec[this_mask:this_mask + mask_len, :] = 0
                return f,t,spec

            elif method == "noise":
                return f,t,spec + np.random.normal(0, fda_th, spec.shape)

            elif method == "none":
                return f,t,augmented
            
            elif method == "rotate":
                from scipy.ndimage import rotate
                angle_list = [-90, -60, -30, 30, 60, 90]
                augmented = rotate(spec, angle_list[fda_th], reshape=False)
                return f,t,augmented

        else:
            if method == "ms-top":
                n_top = param
                k_list = np.argsort(-ms)[:n_top]
                k_idx = k_list[fda_th]
                
                f,t,spec = get_dfs(data[:,k_idx], ms[k_idx], samp_rate=args.sample_rate, window_size=args.default_stft_window, window_step=args.window_step ,agg_type="ms", cache_folder=args.cache_folder)
                augmented = spec
                return f,t,augmented

            elif method == "subband-center":
                n_div = param
                div_len = data.shape[1] // n_div
                k_list = np.arange(0, data.shape[1], div_len) + div_len // 2
                k_idx =  np.argsort(-ms)[k_list[fda_th]]
                f,t,spec = get_dfs(data[:,k_idx], ms[k_idx], samp_rate=args.sample_rate, window_size=self.default_stft_window, window_step=args.window_step ,agg_type="ms", cache_folder=args.cache_folder)
                augmented = spec
                return f,t,augmented
            
            elif method == "ms-even-div":
                n_div = param
                div_len = data.shape[1] // n_div
                k_list = np.arange(0, data.shape[1], div_len)

                k_idx =  np.argsort(-ms)[k_list[fda_th]]

                f,t,spec = get_dfs(data[:,k_idx], ms[k_idx], samp_rate=args.sample_rate, window_size=self.default_stft_window, window_step=args.window_step ,agg_type="ms", cache_folder=args.cache_folder)
                augmented = spec
                return f,t,augmented

            elif method in ["kmeans"]:
                n_kernel = param
                file_key = (file_th, rx_th)
                
                if file_key in self.kmeans_grp:
                    labels = self.kmeans_grp[file_key]
                else:
                    kmeans = KMeans(n_clusters=n_kernel, random_state=0).fit(np.transpose(np.abs(data), (1,0)))
                    labels = kmeans.labels_
                    self.kmeans_grp[file_key] = labels

                # data with labels==fda_th
                data_grp = data[:,labels==fda_th]
                cache_folder = args.cache_folder
                data_str = stringfy_data(data_grp, self.default_stft_window, args.window_step, method + str(file_key))
                hash_data = hashlib.md5(data_str.encode("utf-8")).hexdigest()
                try:
                    # if the file exists, load it
                    data = np.load(cache_folder + hash_data + '.npz')
                    freq_bin = data['freq_bin']
                    ticks = data['ticks']
                    doppler_spectrum = data['doppler_spectrum']
            
                    f_list = freq_bin
                    t_list = ticks
                    augmented = doppler_spectrum
                except:
                    # calculate spec for subcarriers in i-th kernel 
                    for sub_idx in range(data.shape[1]):
                        if labels[sub_idx] == fda_th:
                            f,t,spec = get_dfs(data[:,sub_idx], ms[sub_idx], samp_rate=args.sample_rate, window_size=self.default_stft_window, window_step=args.window_step ,agg_type="ms", cache_folder=args.cache_folder)
                            f_list.append(f)
                            t_list.append(t)
                            ms_this = ms[sub_idx]
                            if ms[sub_idx] < 0:
                                # give it a small number to avoid zeros/negative values in the spec
                                ms_this = 1e-6
                            ms_list.append(ms_this)
                            spectrograms.append(spec)
                    ms_list /= np.sum(ms_list)
                    augmented = np.dot(ms_list, np.array(spectrograms).reshape(np.array(spectrograms).shape[0],-1)).reshape(spec.shape)

                    np.savez(cache_folder + hash_data + '.npz', freq_bin=f, ticks=t, doppler_spectrum=augmented)
            elif method == "subband-ms-top1":
                n_groups = param
                n_per_group = data.shape[1] // n_groups
                k_list = np.arange(0, data.shape[1], n_per_group)
                k_sel = k_list[fda_th]
                k_range = np.arange(k_sel, k_sel+n_per_group)

                k_idx = np.argsort(-ms[k_range])[0]
                
                f,t,spec = get_dfs(data[:,k_idx], ms[k_idx], samp_rate=args.sample_rate, window_size=self.default_stft_window, window_step=args.window_step ,agg_type="ms", cache_folder=args.cache_folder)
                augmented = spec

                return f,t,augmented
            elif method == "subband-group":
                n_groups = param
                n_per_group = data.shape[1] // n_groups
                k_list = np.arange(0, data.shape[1], n_per_group)
                k_sel = k_list[fda_th]
                k_range = np.arange(k_sel, k_sel+n_per_group)

                file_key = (file_th, rx_th)

                # data with labels==fda_th
                data_grp = data[:,k_range]
                cache_folder = args.cache_folder
                data_str = stringfy_data(data_grp, self.default_stft_window, args.window_step, method + str(file_key))
                hash_data = hashlib.md5(data_str.encode("utf-8")).hexdigest()

                try:
                    # if the file exists, load it
                    data = np.load(cache_folder + hash_data + '.npz')
                    freq_bin = data['freq_bin']
                    ticks = data['ticks']
                    doppler_spectrum = data['doppler_spectrum']
            
                    f_list = freq_bin
                    t_list = ticks
                    augmented = doppler_spectrum
                except:
                    for k in k_range:
                        f,t,spec = get_dfs(data[:,k], ms[k], samp_rate=args.sample_rate, window_size=self.default_stft_window, window_step=args.window_step, agg_type="ms", cache_folder=args.cache_folder)
                        f_list.append(f)
                        t_list.append(t)
                        ms_list.append(ms[k])

                        spectrograms.append(spec)
                    ms_list /= np.sum(ms_list)
                    augmented = np.dot(ms_list, np.array(spectrograms).reshape(np.array(spectrograms).shape[0],-1)).reshape(spec.shape)

                    np.savez(cache_folder + hash_data + '.npz', freq_bin=f, ticks=t, doppler_spectrum=augmented)

        return f_list,t_list,augmented


    def space_augment(self, data, ms, method, args=None):
        """
        data: (ts, sub_num)
        args: (sub_num)
        """
        # Use frequency domain augmentation

        return 
