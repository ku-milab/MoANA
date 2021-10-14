from configs.parameter import *
from trainer import Trainer
from tester import Tester
# from tester import Tester

import os
import torch
import numpy as np
from data_loaders import get_data_loader
from data_loaders import get_data_loader_i2c
from torchvision import transforms, datasets
from torch.backends import cudnn
from utils import *

import munch

# Ensure deterministic behavior
SEED = 1
# torch.backends.cudnn.deterministic = True
np.random.seed(hash("setting random seeds") % SEED)
np.random.seed(hash("improves reproducibility") % SEED)
torch.manual_seed(hash("by removing stochasticity") % SEED)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % SEED)

def collate_fn(batch):
    return tuple(zip(*batch))

def main(config):
    _SPLITS = ('train', 'val', 'test')
    _EVAL_METRICS = ['loss', 'classification', 'localization']
    _BEST_CRITERION_METRIC = 'localization'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "ILSVRC": 1000,
        "OpenImages": 100,
    }

    # Device configuration
    # device, device_ids = prepare_device(config.n_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    # organizing data
    dataloaders = get_data_loader(
            data_roots=config.dataset_path,
            metadata_root=config.metadata_path,
            batch_size=config.batch_size,
            workers=config.num_workers,
            resize_size=config.resize_size,
            crop_size=config.crop_size,
            proxy_training_set=config.proxy,
            num_val_sample_per_class=0
        )

    # build model
    if config.mode == "train":
        model = build_train_resnet(config.model_type, _NUM_CLASSES_MAPPING[config.dataset_name], config.parallel, config)
        if config.layer_optim:
            optimizer = torch.optim.SGD([
                {'params': filter(lambda p: p.requires_grad, model.module.conv1.parameters())},
                {'params': filter(lambda p: p.requires_grad, model.module.bn1.parameters())},
                {'params': filter(lambda p: p.requires_grad, model.module.layer1.parameters())},
                {'params': filter(lambda p: p.requires_grad, model.module.layer2.parameters())},
                {'params': filter(lambda p: p.requires_grad, model.module.layer3.parameters())},
                {'params': filter(lambda p: p.requires_grad, model.module.layer4[:3].parameters())},
                {'params': filter(lambda p: p.requires_grad, model.module.layer4[3].parameters()), 'lr': config.lr},
                {'params': filter(lambda p: p.requires_grad, model.module.moana.parameters()), 'lr': config.lr},
                {'params': filter(lambda p: p.requires_grad, model.module.classifier.parameters()), 'lr': config.lr}],
                lr=config.backbone_lr, momentum=config.momentum, weight_decay=config.weight_decay)
        else:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=config.lr_gamma)

        print("============================== " + config.project_name + " Model ==============================")
        if config.wandb:
            wandb.watch(model, criterion, log='all', log_freq=1)
        print(model)
        with open(os.path.join(config.model_save_path, config.project_name + '.txt'), "a") as f:
            print(config, file=f)
        trainer = Trainer(model, dataloaders, criterion, optimizer, scheduler, config)

        trainer.trainval()

    else:
        model = build_test_model(config.model_type, _NUM_CLASSES_MAPPING[config.dataset_name], config.parallel)
        print(model)
        tester = Tester(model, dataloaders, config)
        tester.test()



def mch(**kwargs):
    return munch.Munch(dict(**kwargs))

def configure_data_paths(config):
    train = val = test = os.path.join(config.dataset_path, config.dataset_name)
    data_paths = mch(train=train, val=val, test=test)
    return data_paths

if __name__ == '__main__':
    config = get_parameters()

    config.dataset_path = configure_data_paths(config)
    config.mask_path = os.path.join(config.mask_path, "OpenImages")
    config.metadata_path = os.path.join(config.metadata_path, config.dataset_name)
    if config.mode == 'train':
        config.model_save_path = os.path.join(config.model_save_path, config.project_name)
    config.cam_path = os.path.join(config.cam_path, config.project_name)
    config.cam_test_path = os.path.join(config.cam_test_path, config.project_name)

    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if not os.path.exists(config.cam_path):
        os.makedirs(config.cam_path)
    if not os.path.exists(config.cam_test_path):
        os.makedirs(config.cam_test_path)

    main(config)