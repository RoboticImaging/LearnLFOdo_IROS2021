| Filename | Description |  
| ---------| ----------- |
| [convert_to_png.py](../master/preprocess/convert_to_png.py) | Converts .epirect to png with downsampling.|
| [convert_all_to_png_inter_area.py](../master/preprocess/convert_all_to_png_inter_area.py) | Converts .epirect to png using cv_INTER_AREA interpolation for downsampling.|
| [convert_all_to_png_inter_area_and_augment.py](../master/preprocess/convert_all_to_png_inter_area_and_augment.py) | Same as above but also performs data augmentation.|
| [epi_io.py](../master/preprocess/epi_io.py) | Utilities to work with .epirect images.|
| [generate_video_sequence.py](../master/preprocess/generate_video_sequence.py) | Generates a video sequence for visualization.|
| [process_poses_correctly.py](../master/preprocess/process_poses_correctly.py) | Processes poses recorded by the arm into 4x4 transformation matrices.|