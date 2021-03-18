## Arguments for defining the network and training

### Common arguments

#### Basic parameters

| Argument | Description | Default Value |
| ---------| ----------- | ------------- |
| lfformat | The lightfield format to use `epi`, `focalstack` or `stack` | User needs to specify |
| data | Path to the dataset containing the png images | Users needs to specify |
| name | Name of the experiment. A folder by this name is created in the save-path where output is saved | User needs to specify |
| --save-path `<folder>` | Path to the folder where the output is saved | ~/Documents/checkpoints |

#### Training parameters

| Argument | Description | Default Value |
| ---------| ----------- | ------------- |
| --gray | When this flag is used, only grayscale images are used. | - |
| --sequence-length `<int>` | Number of images to consider in a sample for pose estimation. E.g. if 3 then image at index t is used as the target and images at indices t-1 and t+1 are the reference images. Special: if 2 then only the image at t-1 is used as reference. | 3 |
| --rotation-mode `<str>` | Notation in which the pose estimation network predicts the output. `euler` or `quat` | `euler` |
| --padding-mode `<str>` | Padding mode for image warping. `zeros` for zero-padding or `border` for padding wih the border pixels | `zeros` |
| --epochs `<int>` | Number of epochs for training | 200 |
| -b `<int>` or --batch-size `<int>` | Batch size | 4 |
| --lr `<float>` or --learning-rate `<float>` | Initial learning rate. Currently the learning rate is scaled by 0.1 for every 50 epochs. | 2e-4 |
| --momentum `<float>` | Momentum for SGD. Alpha parameter for ADAM | 0.9 |
| --beta `<float>` | Beta parameter for ADAM | 0.999 |
| --weight-decay `<float>` or --wd `<float>` | Weight decay. NOT USED | 0 |
| --pretrained-disp `<str>` | Path to pretrained Disparity estimation network if it exists | None |
| --pretrained-exppose `<str>` | Path to pretrained Pose estimation network if it exists | None |
| --seed `<int>` | Seed for random functions and network initialization | 0 |
| -p `<float>` or --photo-loss-weight `<float>` | Weight for photometric loss | 1 |
| -m `<float>` or --mask-loss-weight `<float>` | Weight for mask loss | 0 |
| -s `<float>` or --smooth-loss-weight `<float>` | Weight for smoothing loss | 0.1 |
| -g `<float>` or --gt-pose-loss-weight `<float>` | Weight for ground truth pose supervision loss | 0 |

#### Other parameters

| Argument | Description | Default Value |
| ---------| ----------- | ------------- |
| -j `<int>` | Number of data loading workers | 4 |
| --print-freq `<int>`| Print frequency | 10 |
| --log-summary `<str>` | CSV where the per-epoch training and validation stats are saved | progress_log_summary.csv |
| --log-full `<str>` | CSV where the per gradient descent training stats are saved | progress_log_full.csv |
| --log-output | If set then outputs of the disparity estimation network and warped images are logged during validation | - |
| -f `<int>` or --training-output-freq `<int>` | Frequency at which disparity network output and warped images for training at all scaled are output. If 0 then there is no output. | 0 |

### Lightfield format specific parameters

#### Focal stack

| Argument | Description | Default Value |
| ---------| ----------- | ------------- |
| --num-cameras `<int>` | Number of cameras to use to construct the focal stack | User has to specify |
| --num-planes `<int>` | Number of planes the focal stack has to focus on | User has to specify |
| -c `<int>` or --cameras `<int> <int> ..` | List of cameras to use in computing the photometric loss via warping. For singlewarp use only central camera. | 8 |

#### Stack

| Argument | Description | Default Value |
| ---------| ----------- | ------------- |
| -c `<int>` or --cameras `<int> <int> ..` | List of cameras to use in computing the photometric loss via warping. For singlewarp use only central camera. | 8 |
| -k `<str>` or --cameras-stacked `<str>` | If `input` only the input cameras from the -c argument are used to construct the stack. If `full` then all the images are used. | `input` |

Deprecated code for singlewarp training
| Argument | Description | Default Value |
| ---------| ----------- | ------------- |
| -c `<int>` or --cameras `<int> <int> ..` | List of cameras to use to construct the stack | 8 |

#### Epi

| Argument | Description | Default Value |
| ---------| ----------- | ------------- |
| -c `<int>` or --cameras `<int> <int> ..` | In multiwarp training, list of cameras to use in computing the photometric loss via warping. For singlewarp use only central camera. | 8 |
| -e `<str>` or --camera-epi `<str>` | String indicating which of the cameras should be used to form the epipolar image encoding. `vertical`, `horizontal` or `full` | `vertical` |
