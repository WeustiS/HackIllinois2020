import matplotlib.pyplot as plt
from celluloid import Camera
import torch
from baseline import *
from modules import *

def gen_gif(data):
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(len(data)):
        plt.imshow(data[i].permute(1, 2, 0))
        camera.snap()
    return camera.animate()

model = torch.load("./model.pth").to("cuda")
model(torch.load("frame365").to("cuda").float())