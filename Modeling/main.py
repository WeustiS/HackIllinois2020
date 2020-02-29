import baseline as b
import capture_data as d
import visualize as v
import torch

#Import data
data = d.get_frames('vid.mp4').to('cuda:0')

#Declare model
model = b.Baseline('cuda:0').to('cuda:0')

model.do_train(data, data, 100)
model = torch.load('')

real = v.gen_gif(data[:100].detach().cpu())
real.save('real.gif')

out = model.forward(data)

recon = v.gen_gif(out[:100].detach().cpu())
recon.save('recon.gif')



