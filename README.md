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

Our trained model is provided [here](https://drive.google.com/file/d/1g_189B6AFwXWf0L0RJSsBNjP1Y2tztc5/view?usp=sharing). 

## ToF FlyingThings3D dataset

You can download the dataset [here](https://drive.google.com/open?id=1zfOHZqdTPyZr9QDPSRm-Ru3mUVL4zqbQ). The _loader.py_ file is responsible for loading this dataset. 

-------------------
For the most updated details about this work, please refer to the [arxiv paper](). If you find this work useful, please cite

```


```

