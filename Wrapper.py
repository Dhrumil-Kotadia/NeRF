import torch
import cv2

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Torchdata
import matplotlib.pyplot as plt

from FetchData import Nerf_Dataset
from Network import NeRF
from Nerf_Train import Training_Step
from tqdm import tqdm
from Nerf_Train import GetModel
import time
import argparse

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



# NeRF Main
def main():
    """
    Main function to train the NeRF model.
    """
    parser = argparse.ArgumentParser(description='Process model parameters.')

    # Add arguments
    parser.add_argument('--Depth', type=int, default=8, help='Depth of the model.')
    parser.add_argument('--Width', type=int, default=256, help='Width of the model.')
    parser.add_argument('--input_size', type=int, default=3, help='Input size.')
    parser.add_argument('--output_ch', type=int, default=4, help='Output channel size.')
    parser.add_argument('--num_samples', type=int, default=256, help='Number of samples.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')

    # Parse arguments
    args = parser.parse_args()

    # Hyperparameters
    Depth = args.Depth
    Width = args.Width
    input_size = args.input_size
    output_ch = args.output_ch
    num_samples = args.num_samples
    batch_size = args.batch_size
    epochs = args.epochs

    path = '/Datasets/lego/transforms_train.json'
    dataset = Nerf_Dataset(path, im_dim = (200,200))
    train_loader = Torchdata.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    Number_of_Rays = len(train_loader)
    
    # Create NeRF model
    model = NeRF(Depth, Width, input_size, output_ch).to(device)
    
    # Load previous Checkpoint
    # Model_path = '/home/dhrumil/WPI/Sem2/Computer_Vision/Project2/Nerf/Code/Low_Res/Ship/Trained_Model/model_epoch_3.pt'
    # checkpoint = torch.load(Model_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create NeRF Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=450000, gamma=0.1)

    Nerf_Loss = torch.nn.MSELoss()
    Epoch = 0
    
    #############################################################################################################
    while Epoch < epochs:
        batch_time = time.time()
        
        with tqdm(total=Number_of_Rays, unit='rays') as pbar:
            for i, data in enumerate(train_loader, 0):
                Rays = data['Rays'].to(device)
                Images = data['Images'].to(device)
                Rendered = Training_Step(Rays,Images,Num_SamplePoints=num_samples, model = model)
                Rendered_Image = Rendered['rgb_map']
                Loss = Nerf_Loss(Rendered_Image, Images)
                Rendered_Image.detach()
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()
                scheduler.step()
                Loss = Loss.item()
                pbar.update()
                
        torch.save({'epoch': Epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, '/home/dhrumil/WPI/Sem2/Computer_Vision/Project2/Nerf/Code/Checkpoints/' + f'model_epoch_{Epoch+4}.pt')
        Epoch = Epoch + 1
        delta_time_batch = time.time() - batch_time
        tqdm.write(f"================== End of Training, Duration : {delta_time_batch} =================")
    ############################################################################################################
    
    print('Finished Training')


if __name__ == '__main__':
    main()