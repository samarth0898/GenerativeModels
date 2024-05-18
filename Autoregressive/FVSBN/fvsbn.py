"""
Implementation of the fully visible sigmoid belief network (FVSBN)

Autoregressive generative model which, which models P(Xi | Xj, j<i). Models each stage with a simple linear model, 
and sigmoid non-linearity. 

Samples generated on MNSIT 
"""
import torch
import torch.nn as nn

if torch.cuda.is_available():
        device = 'cuda'
if torch.backends.mps.is_available():
    device = 'mps'

class FVSBN(nn.Module):
    def __init__(self, n_dim = 764):
        super(FVSBN, self).__init__()
        self.n_dim  = n_dim
        self.linear_layers = nn.ModuleList(nn.Linear(in_features = max(1, i), out_features = 1) for i in range(n_dim))

    def forward(self, x):
        
        original_size = x.shape 
        x = x.view(original_size[0], -1)
        output = [self.linear_layers[0](torch.zeros(original_size[0],1, device = device))]
        
        for i in range(1, self.n_dim):
            output.append(self.linear_layers[i](x[:,:i]))
    
        return output

        
    def sample(self, x, output_dim):
        """
        Sequentially go through every single sample
        
        """
        sampled_image = torch.zeros(output_dim).to(device)
        sampled_image[0] = x
        
        for pixel in range(1,784):
            
            new_pixel = sampled_image.matmul(self.weights[:,pixel]).to(device)
            sampled_image[pixel] = new_pixel
            
        return sampled_image