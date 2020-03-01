import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torchvision.models import densenet201
from baseline import *
from modules import *
import time

def printing(model):
    print(
        "Sparsity in conv1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv1.conv.weight == 0))
            / float(model.conv1.conv.weight.nelement())
        )
    )
    print(
        "Sparsity in conv2.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv2.conv.weight == 0))
            / float(model.conv2.conv.weight.nelement())
        )
    )
    print(
        "Sparsity in conv3.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv3.conv.weight == 0))
            / float(model.conv3.conv.weight.nelement())
        )
    )

    print(
        "Sparsity in deconv2.weight: {:.2f}%".format(
            100. * float(torch.sum(model.deconv2.deconv.weight == 0))
            / float(model.deconv2.deconv.weight.nelement())
        )
    )
    print(
        "Sparsity in deconv3.weight: {:.2f}%".format(
            100. * float(torch.sum(model.deconv3.deconv.weight == 0))
            / float(model.deconv3.deconv.weight.nelement())
        )
    )
    print(
        "Sparsity in deconv4.weight: {:.2f}%".format(
            100. * float(torch.sum(model.deconv4.deconv.weight == 0))
            / float(model.deconv4.deconv.weight.nelement())
        )
    )

r = torch.rand(1, 3, 1920, 1080)
r0 = torch.rand(1, 3, 1920, 1080)

def measure(model, r):
    arr = [0]*9
    for i in range(50):
        model(r)
    for i in range(10):
        arr = [sum(x) for x in zip(arr, model(r))]
    return [x/(sum(arr)) for x in arr], sum(arr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Baseline()

print("Unpruned time", measure(model, r), "\n")

parameters_to_prune = (
    (model.conv1.conv, 'weight'),
    (model.conv1.conv, 'bias'),
    (model.conv2.conv, 'weight'),
    (model.conv2.conv, 'bias'),
    (model.conv3.conv, 'weight'),
    (model.conv3.conv, 'bias'),
    #(model.conv4.conv, 'weight'),
    #(model.conv4.conv, 'bias'),
    #(model.deconv1.deconv, 'weight'),
    #(model.deconv1.deconv, 'bias'),
    (model.deconv2.deconv, 'weight'),
    (model.deconv2.deconv, 'bias'),
    (model.deconv3.deconv, 'weight'),
    (model.deconv3.deconv, 'bias'),
    (model.deconv4.deconv, 'weight'),
    (model.deconv4.deconv, 'bias'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)
print("Pruned time", measure(model, r))
printing(model)
print()
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.8,
)
print("Pruned time", measure(model, r))
printing(model)