import os
import json
import glob
import cv2
import math

import numpy as np
import torch
import torch.nn as nn

import imageio as iio

import dataset
import model

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, ckpt_dir='./checkpoint', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.ckpt_dir = ckpt_dir
        self.trace_func = trace_func

    def __call__(self, val_loss, model, optim, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optim, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} \n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optim, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optim, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ... \n')
            
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        torch.save({'net': model.state_dict(),
                'optim': optim.state_dict()},
                "%s/model_epoch%d.pth" % (self.ckpt_dir, epoch))

        self.val_loss_min = val_loss

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def off_bound_check(canvas_shape, coord, grid=10):
  # imageio의 프레임은 차원이 (h, w, c)인데 입력되는 좌표는 차원이 (w, h)로 구성됨.
  if np.abs(coord[0]) < canvas_shape[1] // grid or np.abs(coord[0]) > canvas_shape[1] // grid * (grid-1):
    h_off = True
  else:
    h_off = False
  
  if np.abs(coord[1]) < canvas_shape[1] // grid or np.abs(coord[1]) > canvas_shape[0] // grid * (grid-1):
    w_off = True
  else:
    w_off = False
  
  return h_off or w_off

def draw_body_box(canvas, coord, stickwidth=3, radius=10, dots=True, sticks=True):
  limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], \
            [10, 11], [12, 13], [13, 14], \
            [3, 9], [6, 12], [9, 12]]

  colors = [[210, 44, 230], [195, 41, 223], [175, 39, 211], [168, 38, 209], [158, 35, 203], [156, 35, 200], [137, 32, 194], \
            [132, 30, 191], [122, 28, 186], [119, 28, 185], [111, 26, 179], [98, 24, 173], [95, 23, 171], [80, 18, 163], \
            [77, 18, 167], [77, 18, 167], [77, 18, 167], [77, 18, 167]]

  # 스켈레톤의 점을 렌더합니다.
  if dots:
    for i in range(14):
      x, y = coord[i,:]
      if not off_bound_check(canvas.shape, coord[i,:]):
        # 마커의 인덱스가 0일 때(코 마커에서) 타원으로 얼굴 형태를 렌더합니다.
        if i == 0:
          # 인덱스 0인 코 마커를 중심으로 원형의 얼굴을 렌더하는 구문
          cv2.circle(canvas, (int(x), int(y)), radius, colors[i], thickness=stickwidth)
        else:
          cv2.circle(canvas, (int(x), int(y)), stickwidth+1, colors[i], thickness=-1)

  # 스켈레톤의 선을 렌더합니다.
  if sticks:
    for i in range(13):
      index = np.array(limbSeq[i]) - 1
      if off_bound_check(canvas.shape, coord[index[0].astype(int),:]):
        continue

      if not off_bound_check(canvas.shape, coord[index[1].astype(int), :]):
        cur_canvas = canvas.copy()
        Y = coord[index.astype(int), 0]
        X = coord[index.astype(int), 1]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

  return canvas

def train(args):
    if args.mode=="train":
        device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
        
        ckpt_dir = args.ckpt_dir
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        dataset_full = dataset.Dataset(data_dir=args.data_dir, type="train")

        batch_size = args.batch_size

        num_data = len(dataset_full)
        num_data_train = num_data // 100 * (100 - args.split_rate)
        num_batch_train = np.ceil(num_data_train / batch_size)

        dataset_train, dataset_val = torch.utils.data.random_split(dataset_full, [num_data_train, num_data-num_data_train])
        loader_train = torch.utils.data.DataLoader(dataset_train,
                                batch_size=batch_size,
                                shuffle=True, num_workers=0)
        loader_val = torch.utils.data.DataLoader(dataset_val,
                                batch_size=batch_size,
                                shuffle=True, num_workers=0)

        net = model.ResNetRegressor(out_channels=2).to(device)
        criterion = nn.MSELoss(reduction="mean").to(device)
        optim = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0.5, 0.999))

        init_epoch = 0
        early_stop = EarlyStopping(ckpt_dir=ckpt_dir, trace_func=print)
        for epoch in range(init_epoch + 1, 50):
            net.train()
            loss_train = []
            val_data = next(iter(loader_val))
            val_input = val_data["image"].to(device)
            val_target = val_data["label"].to(device)

            for data in enumerate(loader_train, 1):
                image = data["image"].to(device)
                label = data["label"].to(device)

                output = net(image)

                set_requires_grad(net, True)
                optim.zero_grad()

                loss = criterion(output, label)
                loss.backward()
                optim.step()

                loss_train += [float(loss.item())]

            print("TRAIN: EPOCH %04d / %04d | "
                "LOSS %.8f | \n"%
                (epoch, 50, np.mean(loss_train)))
                
            # forward netP
            with torch.no_grad():
                net.eval()
                val_output = net(val_input)
                
                # Early stop when validation loss does not reduce
                val_loss = criterion(val_output, val_target)
                early_stop(val_loss=val_loss, model=net, optim=optim, epoch=epoch)
                
            if early_stop.early_stop:
                break

def evaluate(args):
    if args.mode=="test":
        device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
        
        ckpt_dir = args.ckpt_dir
        if not os.path.exists(ckpt_dir):
            print("Cannot find trained model.")
            return
        
        dataset_dir = os.path.join(os.getcwd(), args.data_dir)
        json_dir = glob.glob(os.path.join(dataset_dir, args.mode, "*.json"))[0]
        with open(json_dir, "r", encoding = "UTF-8-SIG") as file_read:
            test_pose_dict = json.load(file_read)

        test_pose = dataset.filtering_process(dataset.dict2array(test_pose_dict))

        dataset_test = dataset.Dataset(data_dir=args.data_dir, type="test")
        loader_test = torch.utils.data.DataLoader(dataset_test,
                                batch_size=1,
                                shuffle=True, num_workers=0)

        net = model.ResNetRegressor(out_channels=2).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0.5, 0.999))

        epoch, net, optim = model.load_model(args=args,
                                            net=net,
                                            optim=optim)

        with torch.no_grad():
            net.eval()

            evals = []

            for batch_idx, data in enumerate(loader_test):
                image = data["image"].to(device)

                output = net(image)
                predict = output.to("cpu").detach().numpy()
                print(predict.shape)
                predict[0] *= predict[0] * (368/2) + 368
                predict[1] *= predict[1] * (640/2) + 640

                test_pose[0, batch_idx, :] = predict

        result_dir = os.path.join(os.getcwd(), "result")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        title = os.path.splitext(os.path.basename(json_dir))[0]
        writer = iio.get_writer(os.path.join(result_dir, title + ".mp4"), fps=33)
        for index in range(test_pose.shape[1]):
            canvas = np.zeros((368, 640, 3))
            canvas.fill(255)

            paint = draw_body_box(canvas, test_pose[:, index, :2])
            writer.append_data(paint)
        writer.close()