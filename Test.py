import torch
import cv2

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Torchdata
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms as T

from FetchData import Nerf_Dataset
from Network import NeRF
from Nerf_Train import Training_Step
from tqdm import tqdm
from Nerf_Train import GetModel
import time
from PIL import Image

import glob
import contextlib

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def cast_to_depth_image(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_

def save_img(imgs, outfile):
    imgs = imgs / 2 + 0.5
    imgs = torchvision.utils.make_grid(imgs)
    torchvision.utils.save_image(imgs.clone(), outfile, nrow=8)

# NeRF Main
def main():
    
    # Hyperparameters
    Depth = 8
    Width = 256
    input_size = 3
    output_ch = 4
    num_samples = 1000
    batch_size = 32
    epochs = 100
    
    # Create NeRF
    model = NeRF(Depth, Width, input_size, output_ch).to(device)
    Model_path = '/Low_Res/Lego/Trained_Model/WO_PE/model_epoch_8.pt'   # Update the Checkpoint Path Here
    checkpoint = torch.load(Model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    
    Test_Path = '/Datasets/lego/transforms_test.json'   # Update the Test Set Path Here
    Test_Dataset = Nerf_Dataset(Test_Path, im_dim = (200,200))
    Test_Loader = Torchdata.DataLoader(Test_Dataset, batch_size=1000, shuffle=False)
    Number_of_Rays = len(Test_Loader)
    Output = []
    Im_Flag = False
    Generated_Images = []
    for i, data in enumerate(Test_Loader, 0):
        Rays = data['Rays'].to(device)
        Images = data['Images'].to(device)
        Rendered = Training_Step(Rays,Images,Num_SamplePoints=128, model = model)
        Rendered_Image = Rendered['rgb_map']
        Output.append(Rendered_Image.cpu().detach())
        if (i+1)%40 == 0 and i != 0:
            Generated_Images.append(Output)
            Output = []
        # print(i)
    
    i = 0
    for G in Generated_Images:
        for Im in G:
            if Im_Flag == False:
                Rendered_Output = Im
                Im_Flag = True
            else:
                Rendered_Output = torch.cat([Rendered_Output,Im], dim=0)

        Im = Rendered_Output.view(200,200,3).cpu()
        Im2 = Im.permute(2,0,1)
        # save_img(Im2, '/home/dhrumil/WPI/Sem2/Computer_Vision/Project2/Nerf/Code/Low_Res/Lego/Generated_Images/CKPT_4/Rendered_Image_'+str(i)+'.png')
        
        fig, ax = plt.subplots()
        ax.imshow(Im, aspect='auto')  # Display the image data
        ax.axis('off')  # Turn off the axes

        # Remove padding and margins from the figure and axes.
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,hspace=0, wspace=0)
        plt.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        # Save the figure with no padding. bbox_inches='tight' attempts to fit the bounds of the figure to the content.
        plt.savefig('/Low_Res/Lego/Generated_Images/WO_PE/CKPT_5/Rendered_Image_'+str(i)+'.png', bbox_inches='tight', pad_inches=0, dpi=300)
        
        # plt.figure()
        # plt.imshow(Im)
        # plt.imsave('/home/dhrumil/WPI/Sem2/Computer_Vision/Project2/Nerf/Code/Low_Res/Lego/Generated_Images/PLT/CKPT_4/Rendered_Image_'+str(i)+'.png',Im)
        i = i + 1
        Im_Flag = False
        


    # Rendered_Output = Rendered_Output.cpu().detach()
    # Rendered_Output = Rendered_Output.view(800,800,3)
    # Rendered_Output = Rendered_Output.numpy()

    # plt.figure()
    # plt.imshow(Im)
    # plt.show()

    print('Done')

    
    # Create a GIF of the Generated Output
    # filepaths
    fp_out = "/Low_Res/WO_PE_Lego.gif"
    fp_in = []

    for i in range(200):
        fp_in.append("/Low_Res/Lego/Generated_Images/WO_PE/CKPT_5/Rendered_Image_"+str(i)+".png")

    with contextlib.ExitStack() as stack:
        imgs = (stack.enter_context(Image.open(f)) for f in fp_in)
        img = next(imgs)
        img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=50, loop=0)

if __name__ == '__main__':
    main()