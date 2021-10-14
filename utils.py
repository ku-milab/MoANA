import os
import torch
import json
import numpy as np
# from torch.autograd import Variable
# from networks import resnet, senet, gcnet, nlnet, genet
from models import resnet
from models import vgg
# from models import vgg
# from models import inceptionv3

from pathlib import Path
import torch.nn as nn

def t2n(t):
    return t.detach().cpu().numpy().astype(np.float)

def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if scoremap.dtype != np.float:
        raise TypeError("Scoremap must be of np.float type; it is of {} type."
                        .format(scoremap.dtype))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                         .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                         "scoremap.min()={}, scoremap.max()={}."
                         .format(scoremap.min(), scoremap.max()))

def check_box_convention(boxes, convention):
    """
    Args:
        boxes: numpy.ndarray(dtype=np.int or np.float, shape=(num_boxes, 4))
        convention: string. One of ['x0y0x1y1', 'xywh'].
    Raises:
        RuntimeError if box does not meet the convention.
    """
    if (boxes < 0).any():
        raise RuntimeError("Box coordinates must be non-negative.")

    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, 0)
    elif len(boxes.shape) != 2:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if boxes.shape[1] != 4:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if convention == 'x0y0x1y1':
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    elif convention == 'xywh':
        widths = boxes[:, 2]
        heights = boxes[:, 3]
    else:
        raise ValueError("Unknown convention {}.".format(convention))

    if (widths < 0).any() or (heights < 0).any():
        raise RuntimeError("Boxes do not follow the {} convention."
                           .format(convention))

def make_folder(path, version):
        if not os.path.exists(os.path.join(path, version)):
            os.makedirs(os.path.join(path, version))

def improved(best_model, top1_loc, loss):
    if best_model['top1_loc'] < top1_loc:
        improve = True
    elif best_model['top1_loc'] == top1_loc:
        if best_model['loss'] <= loss:
            improve = True
        else:
            improve = False
    else:
        improve = False
    return improve

def weights_init(m):
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.1)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.1)

def build_train_resnet(net_name, num_cls, parallel=False, args=None):
    # Device configuration

    if parallel:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if net_name == "resnet":
        model = resnet.resnet50(net_name, pretrained=True).to(device)
        # num_ftrs = model.fc.in_features
        # model.fc = torch.nn.Linear(num_ftrs, num_cls).to(device)

    elif net_name == "moana":
        model = resnet.resnet50(net_name, num_classes=1000, pretrained=True).to(device)

    if parallel:
        model = torch.nn.DataParallel(model)

    return model

def build_test_model(net_name, num_cls, parallel=False):
    # Device configuration
    if parallel:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model = resnet.pretrainnet50(net_name, pretrained=True, parallel=parallel, num_classes=1200).to(device)
    model = resnet.pretrainnet50(net_name, pretrained=True, parallel=parallel, num_classes=200).to(device)
    return model

def prepare_device(n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle)

def make_result_dict(split, config):
    if config.dataset_name in ('CUB',"CAR","AIR","DOG", 'ILSVRC'):
            result_dict = {
                'train': {
                    'loss': np.inf,
                    'acc': 0,
                },
                'val': {
                    'loss': np.inf,
                    'acc': 0,
                    'loc': 0,
                    'loc_iou': {
                        delta: 0 for delta in config.iou_threshold_list
                    }
                },
                'test': {
                    'loss': np.inf,
                    'acc': 0,
                    'loc': 0,
                    'loc_iou': {
                        delta: 0 for delta in config.iou_threshold_list
                    },
                    'top1_loc': 0,
                    'top5_loc': 0,
                    'gt_loc': 0,
                    'Threshold': 0
                }
            }
    elif config.dataset_name in ('OpenImages'):
        if split == 'test':
            result_dict = {
                'test': {
                    'acc': 0,
                    'loc': 0,
                    'loc_iou': {
                        delta: 0 for delta in config.iou_threshold_list
                    }
                }
            }

        else:
            result_dict = {
                'train': {
                    'loss': np.inf,
                    'acc': 0,
                },
                'val': {
                    'loss': np.inf,
                    'acc': 0,
                    'loc': 0,
                    'loc_iou': {
                        delta: 0 for delta in config.iou_threshold_list
                    }
                }
            }

    return result_dict

def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict

def replace_layer(state_dict, keyword1, keyword2):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword1 in key:
            new_key = key.replace(keyword1, keyword2)
            state_dict[new_key] = state_dict.pop(key)
    return state_dict

def initialize_weights(modules, init_mode):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if init_mode == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif init_mode == 'xavier':
                nn.init.xavier_uniform_(m.weight.data)
            else:
                raise ValueError('Invalid init_mode {}'.format(init_mode))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

def normalize_tensor(x):
    channel_vector = x.view(x.size()[0], x.size()[1], -1)
    minimum, _ = torch.min(channel_vector, dim=-1, keepdim=True)
    maximum, _ = torch.max(channel_vector, dim=-1, keepdim=True)
    normalized_vector = torch.div(channel_vector - minimum, maximum - minimum)
    normalized_tensor = normalized_vector.view(x.size())
    return normalized_tensor

def get_attention(feature, label):
    feat_map = feature.detach().clone()
    attention = feat_map.requires_grad_(True)[
                range(feat_map.size(0)), label, :, :]
    attention = attention.unsqueeze(1)
    attention = normalize_tensor(attention)
    return attention

def correct_topk(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        result = []

        for k in topk:
            correct_k = correct[:k].float().sum(0, keepdim=True)
            result.append(correct_k[0])

        return result

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res