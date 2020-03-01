import baseline as b
import capture_data as d
import visualize as v
import torch

#Declare model
model = b.Baseline('cuda:0').to('cuda:0')

# checkpoint = torch.load('checkpoint.pth')
checkpoint = torch.load('checkpoint.pth')

# model.do_train_on_vid('training_data/vid1', 100000, batch_size=4, checkpoint=checkpoint)
model.do_train_on_vid('training_data/vid1', 100000, batch_size=8, checkpoint=checkpoint)


