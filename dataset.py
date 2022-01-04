import os
import glob
import json

import imageio as iio
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

import torch
import torchvision

def integrate(candidate, subset):
    result = np.zeros((18, len(subset), 2))
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            result[i,n,:] = candidate[index][0:2]
    
    return result

def swapjoint(res_array):
    ret_array = res_array.copy()

    # 마커 3, 4, 5가 6, 7, 8에 비해 오른쪽에 있으면 서로 위치를 바꿉니다.
    if res_array[2, 0, 0] > res_array[5, 0, 0]:
        ret_array[2, 0, 0] = res_array[5, 0, 0]
        ret_array[5, 0, 0] = res_array[2, 0, 0]
    if res_array[3, 0, 0] > res_array[6, 0, 0]:
        ret_array[3, 0, 0] = res_array[6, 0, 0]
        ret_array[6, 0, 0] = res_array[3, 0, 0]
    if res_array[4, 0, 0] > res_array[7, 0, 0]:
        ret_array[4, 0, 0] = res_array[7, 0, 0]
        ret_array[7, 0, 0] = res_array[4, 0, 0]

    # 마커 9, 10, 11이 12, 13, 14에 비해 오른쪽에 있으면 서로 위치를 바꿉니다.
    if res_array[8, 0, 0] > res_array[11, 0, 0]:
        ret_array[8, 0, 0] = res_array[11, 0, 0]
        ret_array[11, 0, 0] = res_array[8, 0, 0]
    if res_array[9, 0, 0] > res_array[12, 0, 0]:
        ret_array[9, 0, 0] = res_array[12, 0, 0]
        ret_array[12, 0, 0] = res_array[9, 0, 0]
    if res_array[10, 0, 0] > res_array[13, 0, 0]:
        ret_array[10, 0, 0] = res_array[13, 0, 0]
        ret_array[13, 0, 0] = res_array[10, 0, 0]
    
    return ret_array

def dict2array(pose_dict, index_list=None):
    res_array = np.zeros((18,1,2))
    count = 0
    for frame in pose_dict["frames"]:
        if len(frame["subset"]) != 1:
            continue

        if index_list == None:
            subset_index = 0
        else:
            subset_index = index_list[0]

        if count == 0:
            result = integrate(frame["candidate"], frame["subset"])
            result = swapjoint(result)
            res_array += result[:, subset_index, :].reshape(18, 1, 2)
            # print(f"count {count}")
            # print(f"{result.shape}")
            count+=1
        else:
            # print("")
            result = integrate(frame["candidate"], frame["subset"])
            result = swapjoint(result)
            res_array = np.concatenate((res_array.copy(), result[:, subset_index, :].reshape(18, 1, 2)), axis=1)
            # print(f"count {count}")
            # print(f"{result.shape}")
            count+=1

    return res_array

def interp_missing(sequence, frame_len):
    x = np.arange(len(sequence))
    idx = np.where(sequence != 0)
    if not len(idx[0]) < 2:
        f = interp1d(x[idx], sequence[idx], fill_value="extrapolate")
        sequence = f(x)

    idx1 = np.where(sequence >= frame_len)
    if not len(idx1[0]) < 2:
        f1 = interp1d(x[idx1], sequence[idx1], fill_value="extrapolate")
        sequence = f1(x)
    
    idx2 = np.where(sequence < 0)
    if not len(idx2[0]) < 2:
        f2 = interp1d(x[idx2], sequence[idx2], fill_value="extrapolate")
        sequence = f2(x)
    
    return sequence

def s_filter(sequence):
    filtered = savgol_filter(sequence, 7, 3, mode="nearest")

    return filtered

def filtering_process(arr):
  blank = np.zeros((368, 640))
  for i in range(18):
    # 결측값을 보간합니다. 성분이 0으로 표기 되어있는 경우에만 결측값으로 판단합니다.
    arr[i,:,0] = interp_missing(arr[i,:,0], blank.shape[1])
    arr[i,:,1] = interp_missing(arr[i,:,1], blank.shape[0])

    filtered = np.zeros_like(arr[i, :, :])
    filtered[:, 0] = s_filter(arr[i,:,0])
    filtered[:, 1] = s_filter(arr[i,:,1])
    if "fil_array" in vars():
        fil_array = np.concatenate((fil_array.copy(), filtered.reshape(1, arr.shape[1], arr.shape[2])), axis=0)
    else:
        fil_array = filtered.reshape(1, arr.shape[1], arr.shape[2]).copy()
  
  return fil_array

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir : str = "./datasets", type : str = "train"):
        dataset_dir = os.path.join(os.getcwd(), data_dir)
        json_dirs = glob.glob(os.path.join(dataset_dir, type, "*.json"))

        pose_list = []
        for json_dir in json_dirs:
            with open(json_dir, "r", encoding="UTF-8-SIG") as f:
                pose_dict = json.load(f)
            
            pose_data = filtering_process(dict2array(pose_dict))
            pose_list.append(pose_data)
        raw_array = np.concatenate(pose_list, axis=1)
        self.data_array = np.concatenate([raw_array[:,:,0].T[np.newaxis,...],
                                          raw_array[:,:,1].T[np.newaxis,...]],
                                         axis=0)

    def bound(self, coord, bound):
        coord = int(coord)

        if coord > bound:
            return bound
        elif coord < -1 * bound:
            return -1 * bound
        else:
            return coord

    def __len__(self):
        return self.data_array.shape[1]

    def __getitem__(self, index):
        data = {}
        blank = np.zeros((3, 368, 640))

        coords = self.data_array[:,index,...]
        for i in range(coords.shape[-1] - 1):
            x_coord = self.bound(coords[0,i], 367)
            y_coord = self.bound(coords[1,i], 639)
            blank[:, x_coord, y_coord] = 1

        data["image"] = torch.from_numpy(blank.astype(np.float32))

        label = self.data_array[:, index, 0]
        label[0] = (label[0] - 368/2) / 368
        label[1] = (label[1] - 640/2) / 640
        data["label"] = torch.from_numpy(label.astype(np.float32))

        return data

if __name__ == "__main__":
    train_dataset = Dataset()
    train_data_0 = next(iter(train_dataset))
    print(train_data_0)