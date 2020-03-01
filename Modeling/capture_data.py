import cv2
import numpy as np
import torch

def get_frames(filepath, max_frames=1e7, verbose=1000):
    

    vidcap = cv2.VideoCapture(filepath)
    success,image = vidcap.read()
    count = 0

    data = []

    while success and count < max_frames:
        # save frame as JPEG file      
        success, image = vidcap.read()
        data.append(image / 255)
        count += 1
        if verbose != -1 and count%verbose==0:
            print("Loading video %s: %.2f%%" % (filepath, count * 100 / max_frames))


    data = np.array(data)
    data = torch.as_tensor(data)
    return data.permute(0, 3, 1, 2)

import numpy as np
def decompose(file_path, save_path, batch_size=64):
    import os
    vidcap = cv2.VideoCapture(file_path)
    success,image = vidcap.read()
    count = 0
    batch_num = 0
    data = torch.zeros(batch_size, image.shape[2], image.shape[0], image.shape[1])
    
    frame_count = 0
    while success and batch_num < batch_size:
        # save frame as JPEG file      
        success, image = vidcap.read()
        image = np.transpose((image / 255), (2, 0, 1))
        data[frame_count] = image
        frame_count += 1
        count += 1
        if count%batch_size==0:
            frame_count = 0
            data = np.array(data)
            torch.save(data, os.path.join(save_path, 'batch_' + str(batch_num) + '.pth'))
            data = torch.zeros(batch_size, image.shape[2], image.shape[0], image.shape[1])
            batch_num += 1
            print("Loading video %s: %.2f%%" % (file_path, batch_num * 100 / batch_size))
            
        print(count)