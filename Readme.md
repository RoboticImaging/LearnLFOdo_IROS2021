## Readme
### Description
This is the code for the paper   
**Unsupervised  Learning  of  Depth  Estimation  and  Visual Odometry  for  Sparse  Light  Field  Cameras**,  
S. Tejaswi Digumarti, Joseph Daniel, Ahalya Ravendran and Donald G. Dansereau,   
under review for submission at IROS 2021.  

**Authors**: [Tejaswi Digumarti](https://tejaswid.github.io), Joseph Daniel    
**Other Contributors**: Ahalya Ravendran, [Donald G. Dansereau](https://roboticimaging.org/)
**Maintainer**: Tejaswi Digumarti  

For further information please see the [Project Website](https://roboticimaging.org/Projects/LearnLFOdo).

### Dependencies
These are required for running the training and inference scripts.  
- [pytorch](https://pytorch.org/get-started/locally/)  
- [numpy](https://numpy.org/install/)  
- [cv2](https://pypi.org/project/opencv-python/)  
- [blessings](https://github.com/erikrose/blessings)  - for handy printing in the terminal  
- [progressbar2](https://github.com/WoLpH/python-progressbar)  - for a progressbar in the terminal  
And the dependencies of these libraries.
  
Optional Dependencies to use the other tools in the repository.  
- [matplotlib](https://matplotlib.org/)  
- [tensorboard](https://www.tensorflow.org/tensorboard)
- [imageio](https://pypi.org/project/imageio/) - Used by DEPRECATED code and not necessary.  
- [evo](https://github.com/MichaelGrupp/evo/tree/60b7927c0838240be87200c444d7dc2949eb44c6)
  This is provided as a submodule.
After cloning the repository do the following in the folder of the repository
```bash
git submodule update --init --recursive
```
Note: If using pycharm as your IDE add external/evo to sources.

### Setup
#### Setting up an anaconda environment for training
An anaconda enviornment for training can be setup using the following commands.
Please follow the sequence, otherwise there may be some inter-dependent library conflicts.  
```bash
conda create -n epienv python=3.7
conda activate epienv
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge blessings progressbar2
conda install -c conda-forge opencv
conda install -c conda-forge matplotlib
conda install -c conda-forge tensorboard
```

### Training
The python file to run for training is the [`train_multiwarp.py`](../master/train_multiwarp.py). 
Example training scripts with configuration parameters are present in 
the [`training_scripts`](../master/training_scripts) folder.

The training process consists of the following steps.  
1. Parse the input arguments to determine the path to the dataset, save folder, lightfield format to use for training 
and a few other configuration parameters. These functions are defined in [`parser.py`](../master/parser.py). A full list
of arguments can be found [here](../master/Arguments.md).

2. Data loaders (from [`epimodule.py`](../master/epimodule.py)) specific to the lightfield format are then
initialized. The lightfield formats are the following.  
   - `focalstack`: A focal stack image formed by layering images from all the cameras of the multi-aperture camera
    and computing the average intensity at each pixel. The plane of focus is determined by the amount by which 
    each image is shifted (vertically or horizontally), before adding it to the image from the central camera.
    The function [`load_multiplane_focalstack`](../master/epimodule.py#L276) does this.
   
   - `stack`: This is nothing but a concatenation of all the images from all the cameras, to get a
    3*N channel (colour) or N channel (grayscale) image where N is the number of cameras.
    The function [`load_stacked_epi`](../master/epimodule.py#L369) does this.
    
   - `epi`: An epipolar plane image (EPI) is formed by taking horizontal or vertical slices of the images from the
   multi-array camera and concatenating these slices together. If vertical slices are taken and stacked horizontally,
    it will result in an image of size (height x N\*width) and if horizontal slices are taken and stacked vertically,
    it will result in an image of size (N\*height x width). 
    The function [`load_tiled_epi`](../master/epimodule.py#L353) does this.

3. Prepare pytorch dataloaders for training and validation using the above dataloaders.  
4. If the lightfield format is `epi`, then encoders [`RelativeEpiEncoder`](../master/lfmodels/EpiEncoder.py#L7) and
[`EpiEncoder`](../master/lfmodels/EpiEncoder.py#L83) are loaded to encode the lightfield image into an image that forms
the input to the Pose estimation network and the Disparity network respectively.  
5. Load the Disparity and Pose estimation networks with pre-trained weights if available.    
6. Train and validate over the specified number of epochs.  

### Inference
Use the script [infer_multiwarp.py](../master/infer_multiwarp.py) to perform inference.
Examples using this script with parameters set are in the [validation_scripts](../master/validation_scripts) folder.

To infer just depth and not pose (e.g. for a single input image), then use [infer_depth](../master/infer_depth.py).

### File Description
A detailed description of all the files can be found [here](../master/file_description.md).
