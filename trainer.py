import time
import wandb
import os
from tqdm import tqdm
from utils import *
from heatmaps import CAMComputer
from modules import cutmix
from modules import has

import torch.nn.functional as F
import pickle
from sklearn.metrics import accuracy_score
import numpy as np

class Trainer(object):
    def __init__(self, model, data_loader, criterion, optimizer, scheduler, config):
        self.config = config

        # get the size of train and evaluation data
        if isinstance(data_loader, dict):
            self.dataset_sizes = {x: len(data_loader[x].dataset) for x in data_loader.keys()}
        self.project_name = config.project_name

        # Device configuration
        # self.device, self.device_ids = prepare_device(config.n_gpu)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.data_loader = data_loader
        self.ce_loss = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.wandb = config.wandb
        self.result = dict()

        # Config
        self.model_type = config.model_type
        self.dataset_name = config.dataset_name
        self.iou_threshold_list = config.iou_threshold_list
        self.mask_path = config.mask_path
        self.metadata_path = config.metadata_path
        self.multi_iou_eval = config.multi_iou_eval

        self.cam_curve_interval = config.cam_curve_interval
        self.multi_contour_eval = config.multi_contour_eval

        # Model hyper-parameters
        self.epoch = config.epoch
        self.batch_size = config.batch_size

        # key: 'acc', 'loss'
        self.best_model = {"top1_cls": -np.inf, "loss": np.inf,
                           "MaxBoxAcc": -np.inf, "top1_loc": -np.inf, "top5_loc": -np.inf, "gt_loc": -np.inf}

        self.model_save_path = config.model_save_path
        self.cam_path = config.cam_path

    def trainval(self):

        # num_example = 0  # number of examples seen
        # batch = 0

        for epoch in range(1, self.epoch+1):
            if self.wandb:
                wandb.log({"epoch": epoch}, step=epoch)
            print("----- Epoch [{} / {}] -----".format(epoch, self.epoch))
            time.sleep(0.1)
            train_running_loss = 0.0
            valid_running_loss = 0.0
            train_corrects = 0
            valid_corrects = 0

            # # ------------------------- Train -------------------------------------
            split = "train"
            self.model.train()
            result = make_result_dict(split, self.config)

            for iter, (data, target, img_ids) in enumerate(tqdm(self.data_loader[split], desc="Train", mininterval=0.1)):
                data, target = data.to(self.device), target.to(self.device)
                # with torch.no_grad():
                with torch.set_grad_enabled(split == 'train'):
                    self.optimizer.zero_grad()

                    # Forward pass
                    output = self.model(data)
                    batch_loss = self.ce_loss(output['logits'], target)

                    # Backward pass
                    batch_loss.backward()

                    # Step with optimizer
                    self.optimizer.step()

                    # Predict Class
                    _, preds = torch.max(output['logits'], 1)

                train_running_loss += batch_loss.item() * data.size(0)
                train_corrects += torch.sum(preds == target.data)

            self.scheduler.step()
            train_loss = train_running_loss / self.dataset_sizes[split]
            train_top1_cls = train_corrects.cpu().numpy()/self.dataset_sizes[split]

            torch.save(self.model.state_dict(),
                       os.path.join(self.model_save_path, self.project_name + '_{}.pth'.format(epoch)))

            result[split]['loss'] = train_loss
            result[split]['acc'] = train_top1_cls

            print("Train:\t[ACC:{:.4f}, Loss:{:.4f}]\n"
                  .format(train_top1_cls, train_loss))

