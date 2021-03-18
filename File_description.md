## File description

| Filename | Description |  
| ---------| ----------- |
| [Readme.md](../master/Readme.md) | Main Readme file. |
| [Arguments.md](../master/Arguments.md) | Description of all the arguments that can be set at run time. |
| [File_description.md](../master/File_description.md) | This file with description of all files. |

### Common code  

| Filename | Description |  
| ---------| ----------- |
| [custom_transforms.py](../master/custom_transforms.py) | Contains custom transforms used during training and some utilities. |
| [epimodule.py](../master/epimodule.py) | Contains functions for loading epipolar images in various formats.|
| [intrinsics.txt](../master/intrinsics.txt) | File with camera intrinsics.|
| [inverse_warp.py](../master/inverse_warp.py) | Implements the inverse warp functions.|
| [logger.py](../master/logger.py) | Contains functions for logging.|
| [loss_functions.py](../master/loss_functions.py) | Different loss functions. |
| [utils.py](../master/utils.py) | Extra utility functions. |

### Code for training and validation

| Filename | Description |  
| ---------| ----------- |
| [parser.py](../master/parser.py) | Utility functions for parsing user input when training scripts are run. |
| [multiwarp_dataloader.py](../master/multiwarp_dataloader.py) | Contains functions for loading datasets as required by pytorch.|
| [train_multiwarp.py](../master/train_multiwarp.py) | Main training script. Works for both multiwarp and singlewarp training. |
| [infer_multiwarp.py](../master/infer_multiwarp.py) | Script for inference of both depth and relative pose.|
| [infer_depth.py](../master/infer_depth.py) | Script for inferring only depth (not pose).|

### lfmodels: 
This folder contains the disparity and pose estimation networks and the encoder networks for 
light field images.

### Other folders:
**training_scripts:** Contains bash scripts to train the networks in different configurations.    
**validation_scripts:** Contains bash scripts for running inference for different configurations.  
**preprocess:** Contains python scripts to preprocess data captured by the epi module.  
**preprocessing_scripts:** Contains bash scripts to automate preprocessing.  
**paper_specific:** Contains python scripts to generate results for the paper.  
**tests:** Testbed folder. 

### Deprecated code:
Code for single warp training only  

| Filename | Description |  
| ---------| ----------- |
| [dataloader.py](../master/dataloader.py) | Contains functions for loading datasets as required by pytorch.|
| [focalstack.py](../master/focalstack.py) | Seems like a repetition of the epimodule.py. Not used.|
| [infer.py](../master/infer.py) | File where prediction happens.|
| [train.py](../master/train.py) | Training routine. |

Code for supervised training of PoseNet  

| Filename | Description |  
| ---------| ----------- |
| [infer_supervised.py](../master/infer_supervised.py) | File where prediction happens - just for PoseNet.|
| [train_supervised.py](../master/train_supervised.py) | training routine for PoseNet. |