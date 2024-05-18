import torch
import numpy as np

import torch
from Network import NeRF



def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    """
    Exclusive cumulative product of a tensor along its last dimension.
    Note: This function is inspired by this NeRF implementation:  https://github.com/murumura/NeRF-Simple
    """
    Out = torch.cumprod(tensor, dim=-1)
    Out = torch.roll(Out, 1, dims=-1)
    Out[..., 0] = 1.0
    return Out

class SimpleVolumeRenderer(torch.nn.Module):
    '''
    Simple Volume Renderer for NeRF.
    This class is inspired from the github repository: https://github.com/murumura/NeRF-Simple
    '''
    def __init__(self,
        train_radiance_field_noise_std=0.0,val_radiance_field_noise_std=0.0,white_background=False,attenuation_threshold=1e-3,device='cuda',**kwargs):
        
        super().__init__()
        
        self.train_radiance_field_noise_std = train_radiance_field_noise_std
        self.val_radiance_field_noise_std = val_radiance_field_noise_std
        self.attenuation_threshold = attenuation_threshold
        self.use_white_bkg = white_background

        E = torch.tensor([1e10]).to(device)
        E.requires_grad = False
        
        self.register_buffer("epsilon", E)
        
    def forward(self, radiance_field, depth_values, ray_directions):
    
        
        radiance_field_noise_std = self.train_radiance_field_noise_std

        delta = torch.cat((depth_values[..., 1:] - depth_values[..., :-1],self.epsilon.expand(depth_values[..., :1].shape),),dim=-1,)
        delta = delta * ray_directions[..., None, :].norm(p=2, dim=-1)

        rgb = radiance_field[..., :3]

        noise = 0.0
        if radiance_field_noise_std > 0.0:
            noise = (
                torch.randn(radiance_field[..., 3].shape, dtype=radiance_field.dtype, device=radiance_field.device) *
                radiance_field_noise_std
            )

        sigs = torch.nn.functional.relu(radiance_field[..., 3] + noise)
        A = 1.0 - torch.exp(-sigs * delta)

        T_i = cumprod_exclusive(1.0 - A + 1e-10)  #(ray_count, num_samples)

        weight = A * T_i  #(ray_count, num_samples)
        masked_weights = (T_i > self.attenuation_threshold).float()  #(ray_count, num_samples)

        rgb = weight[..., None] * rgb  #(ray_count, num_samples, 3)
        rgb = rgb.sum(dim=-2)  #(ray_count, 3)
        acc = weight.sum(dim=-1)  #(ray_count, )

        depth = (weight * depth_values).sum(dim=-1)  # (ray_count, )

        disp = 1.0 / torch.max(1e-10 * torch.ones_like(depth), depth / acc)
        disp[torch.isnan(disp)] = 0

        if self.use_white_bkg:
            rgb = rgb + (1.0 - acc[..., None])

        ret = {'rgb_map': rgb,'depth_map': depth,'weights': weight,'mask_weights': masked_weights,'acc_map': acc,'disp_map': disp}
        return ret

def Get_Sample_Points(Ray_Origins, Ray_Directions, Nears, Fars, Num_SamplePoints):
    """
    Function to generate sample points along the rays.
    Input: Ray_Origins, Ray_Directions, Nears, Fars, Num_SamplePoints
    Output: Sample_Points, Ray_Directions, Point_Intervals
    """
    Intervals = torch.linspace(0., 1., steps=Num_SamplePoints, requires_grad=False).to(Ray_Origins.device)
    Num_Rays = Ray_Origins.shape[0]

    Intervals = Intervals.unsqueeze(0)
    Nears = Nears.unsqueeze(1)
    Fars = Fars.unsqueeze(1)
    
    Point_Intervals = Nears + Intervals * (Fars - Nears)

    #Perturb generated sample points
    Perturb = torch.rand_like(Point_Intervals)
    Perturb = (Perturb - 0.5) * (Fars - Nears) / Num_SamplePoints
    Perturb[:,0] = 0.
    Point_Intervals = Point_Intervals + Perturb
    Point_Intervals = torch.reshape(Point_Intervals, (Num_Rays, -1))

    #Samle points from intervals
    Sample_Points = Ray_Origins.unsqueeze(1) + Ray_Directions.unsqueeze(1) * Point_Intervals.unsqueeze(-1)
    Ray_Directions = Ray_Directions.unsqueeze(1).expand(-1, Num_SamplePoints, -1)
    return Sample_Points, Ray_Directions, Point_Intervals

def Training_Step(Rays, Images, Num_SamplePoints, model):
    """
    Function to perform training step for NeRF.
    Input: Rays, Images, Num_SamplePoints, model
    Output: Rendered Output
    """
    Ray_Origins = Rays[:, 0:3]
    Ray_Directions = Rays[:, 3:6]
    Renderer = SimpleVolumeRenderer()
    Nears = Rays[:, 6]
    Fars = Rays[:, 7]
    Sample_points, Ray_Directions_Exp,Point_Intervals = Get_Sample_Points(Ray_Origins, Ray_Directions, Nears, Fars, Num_SamplePoints)
    Output = model(Sample_points, Ray_Directions_Exp)
    Rendered = Renderer(Output, Point_Intervals, Ray_Directions)
    return Rendered

def GetModel():
    """
    Function to get NeRF model.
    """
    return NeRF()