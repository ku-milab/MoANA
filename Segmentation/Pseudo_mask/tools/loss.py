import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy2d(input, target, weight=None, size_average=True, pad=False):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    if pad:
        target = torch.nn.ZeroPad2d(100)(target)

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.unsqueeze(1)
        target = F.interpolate(target.float(), size=(h, w), mode="nearest").long()
        target = target.squeeze(1)
    elif h < ht and w < wt:  # upsample images
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, ignore_index=250,
        reduction='elementwise_mean'
    )
    return loss


def multi_scale_cross_entropy2d(
    input, target, weight=None, size_average=True, scale_weight=None
):
    if not isinstance(input, tuple): # when evaluation
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to('cuda' if target.is_cuda else 'cpu')

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input,
                                  target,
                                  K,
                                  weight=None,
                                  size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input,
                                   target,
                                   K,
                                   weight=None,
                                   size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input,
                               target,
                               weight=weight,
                               reduce=False,
                               size_average=False,
                               ignore_index=250)
        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)
