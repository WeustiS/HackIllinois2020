import cv2
import numpy as np
import torch

def get_frames(filepath):
    

    vidcap = cv2.VideoCapture(filepath)
    success,image = vidcap.read()
    count = 0

    data = []

    while success and count < 10:
        cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
        success, image = vidcap.read()
        data.append(image / 255)
        count += 1
    data = np.array(data)
    data = torch.as_tensor(data)
    return data.permute(0, 3, 1, 2)
