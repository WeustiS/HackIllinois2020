import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

import modules as m

class Baseline(nn.Module):
    def __init__(self, device):
        super(Baseline, self).__init__()
        
        self.device = device
        
        self.conv1 = m.CNNBlock2d(3, 16, 3, use_wavelet=True)
        self.conv2 = m.CNNBlock2d(16, 32, 3)
        self.conv3 = m.CNNBlock2d(32, 64, 3)
        self.conv4 = m.CNNBlock2d(64, 128, 3)
        
        self.deconv1 = m.TransposedCNNBlock2d(128, 64, 4)
        self.deconv2 = m.TransposedCNNBlock2d(64, 32, 4)
        self.deconv3 = m.TransposedCNNBlock2d(32, 16, 4)
        self.deconv4 = m.TransposedCNNBlock2d(17, 3, 4)
        
    def forward(self, x):
        interm_out, res = self.conv1(x)
        interm_out = self.conv2(interm_out)
        interm_out = self.conv3(interm_out)
        interm_out = self.conv4(interm_out)
        
        interm_out = self.deconv1(interm_out)
        interm_out = self.deconv2(interm_out)
        interm_out = self.deconv3(interm_out)
        interm_out = torch.cat((interm_out, res), 1)
        interm_out = self.deconv4(interm_out)
        
        return interm_out
        
    def encode(self, x):
        interm_out, res = self.conv1(x)
        interm_out = self.conv2(interm_out)
        interm_out = self.conv3(interm_out)
        interm_out = self.conv4(interm_out)
        
        return interm_out

    def do_train(self, x, y, epochs, batch_size=32, lr=1e-4, verbose=1, checkpoint=None):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        if checkpoint:
            self.load_state_dict(checkpoint['state_dict'])
            self.to(self.device)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        min_loss = 1e8
        for epoch in range(epochs):
            #self.train()
            #Shuffle data for stochastic iteration
            shuffle = np.random.permutation(len(x))
            x = x[shuffle]
            y = y[shuffle]
            #Interate through data
            running_loss = 0
            
            for batch_idx in range(len(x) // batch_size):
                xbatch = x[batch_idx*batch_size:(batch_idx+1)*batch_size].to(self.device).float()
                ybatch = y[batch_idx*batch_size:(batch_idx+1)*batch_size].to(self.device).float()
                
                y_hat = self.forward(xbatch)
            
                loss = ((y_hat - ybatch)**2).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                #if not torch.isnan(loss): 
                running_loss += loss.detach()                
            
            running_loss /= (len(x) // batch_size)
            
            if (epoch + 1)%verbose == 0:
                print('[%d]: %.16f' % (epoch + 1, running_loss))
                scheduler.step(running_loss)
                if running_loss < min_loss:
                    min_loss = running_loss
                    #Save network
                    checkpoint = {'state_dict':self.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                        
                    torch.save(checkpoint, 'checkpoint.pth')   

def sparsify(x):
    idx = torch.nonzero(x, as_tuple=True)
    data = x[idx]
    
    return data, idx, data.shape
    
def de_sparsify(x, idx, shape):
    out = torch.zeros(shape)
    out[idx] = x
    
    return out
    
    
    
    
    