import matplotlib.pyplot as plt
from celluloid import Camera

def gen_gif(data):
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(len(data)):
        plt.imshow(data[i].permute(1, 2, 0))
        camera.snap()
    return camera.animate()
    
    