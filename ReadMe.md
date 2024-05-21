# Neural Radiance Fields (NeRF)

## Introduction
This work is our implementation and analysis of Neural Radiance Fields(NeRF). This neural network is a fully connected network with input characterized by spatial positions (x, y, z) and viewing directions (θ, Φ), providing volume density and RGB pixel values corresponding to the viewing direction as outputs. NeRF uses classical volume rendering techniques, considering each point as a ray starting from the camera center and passing through each pixel to the world. The premise of the [paper](https://arxiv.org/abs/2003.08934) involves generating images of a scene from new viewpoints, a problem falling under novel image synthesis.

<p float="left">
 <img src="Images/Lego/Lego.gif" width="800"/>
</p>

## Dependencies

To run the code, install the following necessary packages.

1. PyTorch
2. PIL
3. Numpy
4. tqdm
5. time
6. Matplotlib

## Usage

Follow the steps to run the code:

1. Set the Hyperparameters (Batch_Size, Learning_Rate, Step_Size, Epochs etc.) in Wrapper.py using command line arguments.
2. Set the appropriate paths for Training data in the Wrapper.py and FetchData.py.
3. Run the file Wrapper.py with appropriate arguments.
4. For testing, provide the appropriate paths for trained model, Rendered Image file generation and Gif generation in Test.py.
5. Run Test.py to generate the output.

## Dataset

Download the [dataset](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) used in the official implementation.

## Network
The network implemented is the one mentioned in the official NeRF paper as shown in the following figure. The implementation is a fully connected network and has hidden layers with 256 channels each and a ReLU after it. After 4 layers, there is a skip connection that concatenates the input to the fifth layer. An additional layer outputs the density and is then concatenated with the viewing direction. It is then processed by an additional fully connected layer with 128 channels which gives the RGB output.

<p float="left">
 <img src="Images/Network.png" width="800"/>
</p>
<p align="center"><em>NeRF Network<em></p>

- **Image Resolution for Training:** 200x200
- **Number of samples per ray:** 256
- **Learning rate:** 0.0005
- **Optimizer:** Adam
- **Loss:** MSE loss
- **Near, Far bounds:** 2, 6

## Outputs

The rendered novel views are compared with the ground truth in the following figures.

<p float="left">
 <img src="Images/Lego/GT/r_21.png" width="400"/>
 <img src="Images/Lego/Rendered/Rendered_Image_21.png" width="400"/>
 <img src="Images/Lego/GT/r_62.png" width="400"/>
 <img src="Images/Lego/Rendered/Rendered_Image_62.png" width="400"/>
 <img src="Images/Lego/GT/r_95.png" width="400"/>
 <img src="Images/Lego/Rendered/Rendered_Image_95.png" width="400"/>
</p>
<p align="center"><em>Ground Truth (Left) and Rendered Image(Right)<em></p>

Please check the report for more details.

## Author
- **Dhrumil Kotadia**  
  Robotics Engineering Department, Worcester Polytechnic Institute  
