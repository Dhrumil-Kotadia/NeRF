import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def ExtractJSONData(json_file_path):
    """
    Extracts JSON data from the given file path.
    Args: Path to the JSON file(str).
    Returns: dict: JSON data.
    """
    with open(json_file_path) as f:
        data = json.load(f)
    return data

def GetFrameData(Frame):    
    """
    Extracts Pose and Image for the given Frame.
    Args: Frame(dict): Frame data.
    Returns: Pose(np.array): Pose and Image.
    """

    TM = np.array(Frame['transform_matrix'])
    # R = TM[0:3,0:3]
    # t = TM[0:3,3]
    Pose = TM[:3,:4]
    def_path_string = '/Datasets/lego'
    Im_Path = Frame['file_path']
    Im_Path = Im_Path.replace('.','')
    Im_Path = def_path_string + Im_Path + '.png'
    Im = Image.open(Im_Path)
    Im = Im.resize((200,200), Image.LANCZOS)
    
    return Pose, Im

def Get_Ray_Directions(Width, Height, focal):
    """
    Function to get Ray Directions.
    Args: Width, Height, focal
    Returns: Ray_Directions
    """
    i, j = torch.meshgrid(torch.linspace(0, Width - 1, Width), torch.linspace(0, Height - 1, Height))
    i = i.t()
    j = j.t()
    directions = torch.stack([(i - Width * 0.5) / focal, -(j - Height * 0.5) / focal, -torch.ones_like(i)], dim=-1)
    return directions

def get_rays(Pose, Ray_Directions):
    """
    Function to get Rays.
    Args: Pose, Ray_Directions
    Returns: Ray_Origins, Ray_Directions
    """
    Ray_Directions = Ray_Directions@Pose[:,:3].T
    Ray_Directions = Ray_Directions/torch.norm(Ray_Directions, dim=-1, keepdim=True)

    Ray_Origins = Pose[:,3].expand(Ray_Directions.shape)
    Ray_Directions = Ray_Directions.view(-1,3)
    Ray_Origins = Ray_Origins.view(-1,3)
    return Ray_Origins, Ray_Directions


class Nerf_Dataset(Dataset):
    """
    Dataset class for NeRF.
    Args: path, im_dim
    """
    def __init__(self, path, im_dim = (200,200)):
        self.path = path
        self.width,self.height = im_dim
        self.transform = transforms.ToTensor()
        self.extract_data()
        

    def extract_data(self):
        data = ExtractJSONData(self.path)
        focal = 0.5 * self.width / np.tan(0.5 * data['camera_angle_x'])    # Getting focal length from field of view. Help: https://github.com/yenchenlin/nerf-pytorch/issues/41
        self.near = 2.0
        self.far = 6.0
        self.range = np.array([self.near, self.far])
        self.Ray_Directions = Get_Ray_Directions(self.width, self.height, focal)
        self.rays = []
        self.images = []
        for frame in data['frames']:
            Pose, Im = GetFrameData(frame)
            Im = self.transform(Im)
            Pose = torch.FloatTensor(Pose)
            Im = Im.view(4, -1).permute(1, 0)
            Im = Im[:, :3] * Im[:, -1:] + (1 - Im[:, -1:])
            
            self.images.append(Im)
            Ray_Origin, Ray_Direction = get_rays(Pose, self.Ray_Directions)
            self.rays.append(torch.cat([Ray_Origin, Ray_Direction, self.range[0]*torch.ones_like(Ray_Origin[:, :1]), self.range[1]*torch.ones_like(Ray_Origin[:, :1])], 1))
        self.rays = torch.cat(self.rays, 0)
        self.images = torch.cat(self.images, 0)


    def __len__(self):
        return len(self.rays)
    
    def __getitem__(self, Index):
        return {'Rays':self.rays[Index],'Images':self.images[Index]}
