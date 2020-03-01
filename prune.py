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
        "Sparsity in conv4.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv4.conv.weight == 0))
            / float(model.conv4.conv.weight.nelement())
        )
    )
    print(
        "Sparsity in deconv1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.deconv1.deconv.weight == 0))
            / float(model.deconv1.deconv.weight.nelement())
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

r = torch.rand(1, 3, 1920, 2160).to('cuda')
r0 = torch.rand(1, 3, 1920, 2160).to('cuda')

def measure(model, r):
    t0 = time.time()
    for i in range(25):
        model(r)
    return time.time()-t0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Baseline().to("cuda")

print("Unpruned time", measure(model, r), "\n")

parameters_to_prune = (
    (model.conv1.conv, 'weight'),
    (model.conv1.conv, 'bias'),
    (model.conv2.conv, 'weight'),
    (model.conv2.conv, 'bias'),
    (model.conv3.conv, 'weight'),
    (model.conv3.conv, 'bias'),
    (model.conv4.conv, 'weight'),
    (model.conv4.conv, 'bias'),
    (model.deconv1.deconv, 'weight'),
    (model.deconv1.deconv, 'bias'),
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
print("Pruned time", measure(model, r0))
printing(model)
print()
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.7,
)
print("Pruned time", measure(model, r0))
printing(model)