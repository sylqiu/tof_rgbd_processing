# Deep End-to-End Alignment and Refinement for Time-of-Flight RGB-D modules

This repository contains the TensorFlow (1.2) implementation of the paper "Deep End-to-End Alignment and Refinement for Time-of-Flight RGB-D modules". This implementation has been tested on Ubuntu 14.04. with CUDA 8.0.

To use it, first create your python virtual environment, and install the requirements by
```
pip install -r requirements.txt
```
Then compile the cuda operations: in _/utils_ and _/utils/ops/warp_by_flow_, type in the Terminal
```
make
```
Now you should be ready to go. To test the model, use the _fullmodel.py_ in the _/fullmodel_ directory, where you can set the data directory and output directory.

Our trained model is provided [here](https://drive.google.com/drive/folders/1XASaOfcp3TzQJ0A2fMaXex-0eihha0vg?usp=sharing). 

## ToF FlyingThings3D dataset

You can download the RGB-Depth dataset (~20GB) [here](https://drive.google.com/drive/folders/1XASaOfcp3TzQJ0A2fMaXex-0eihha0vg?usp=sharing). The _loader.py_ file is responsible for loading this dataset. Since the original transient rendering is too large to host on Drive.
_Update 17 March, 2020_ We provide the original blender&pbrt files in the same folder with code about generating the data. 

Note that our data generation follows the protocol from Su et al. Deep End-to-End Time-of-Flight Imaging, CVPR 2018.

Note: In the Drive there is a calib.bin file, which should be used if mvg_aug is set to be true when training. There is another calib.bin file in the test_real folder, which contains a sample real test image. The calib.bin file there is for real calibration.

-------------------
For the most updated details about this work, please refer to the [arxiv paper](https://arxiv.org/abs/1909.07623). If you find this work useful, please cite

```
@inproceedings{qiu2019rgbd,
  title={Deep End-to-End Alignment and Refinement for Time-of-Flight RGB-D modules},
  author={Di Qiu, Jiahao Pang, Wenxiu Sun, Chengxi Yang},
  booktitle={International Conference in Computer Vision (ICCV)},
  year={2019}
}
```
-------------------
Disclaimer: This software and related data are published for academic and non-commercial use only.

