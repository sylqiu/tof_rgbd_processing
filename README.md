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

You can download the RGB-Depth dataset (~20GB) [here](https://drive.google.com/drive/folders/1XASaOfcp3TzQJ0A2fMaXex-0eihha0vg?usp=sharing). The _loader.py_ file is responsible for loading this dataset. 

**Update 17 March, 2020**: Since the original transient rendering is too large to host on Drive, we provide the original blender&pbrt files in the same Drive folder with code on generating the data. 

Note that our data generation follows the protocol from Su et al. Deep End-to-End Time-of-Flight Imaging, CVPR 2018.

## Synthetic ToF Dataset Generation Using Blender and PBRT

### Introduction
This repo contains the source and data files for generating synthetic ToF data.

There are two major parts in the data generation pipeline:
* Produce ground truth depth using __Blender__ with a _.blend_ file.
* Produce transient renderings using __pbrt-v3-tof__ with a (or multiple) _.pbrt_ file(s), and use MatLab to process the renderings as ToF raw correlation measurements or ToF depth images (no plane correction).

Optionally, you may render corresponding color images using Blender or the official version of pbrt.

The major files in repo is organized as follows:

```

|- blender_utils
    |-- export_path.py # export camera locations in Blender's Timeline, should be run inside of Blender #
    |-- output_zpass.py # python script for writing ground truth depth #
    |-- lighting_multiple_output # further applications for use of python in blender #

|- pbrt-v3-tof
    |-- example 
        |-- batch_run_pbrt.sh # pbrt rendering example #

|- transient_processing
    |-- example
        |-- transient_to_depth.m # MatLab script for converting transient rendering into ToF correlation measurements and ToF depth images #

|- pbrt_material_augmentation
    |-- exanple
        |-- output_materail.m # MatLab script for writing material library .pbrt files #

|- scenes # 3D models and camera paths # 
 
```

### Installation pre-requisite

Tested under Ubuntu 14.04
* MatLabEXR: <https://blog.csdn.net/lqhbupt/article/details/7867697> (in Chinese)
* pbrt-v3-tof: This is a costomized renderer for transient rendering provided by Su et al. <https://vccimaging.org/Publications/Su2018EndToEndTOF/>, ordinary cmake build should work. For official pbrt, see <https://www.pbrt.org/>
* Blender 2.7: <https://www.blender.org/> 


### Workflow

1. Render ground truth depth images (no plane correction) using the Depth Pass in the Blender's Cycles renderer. In Blender's GUI, it is visible in the Node editor's Renderlayers viewport. 
Make sure the "Use Nodes" is ticked. Set your camera position and hit the camera-shot button to render the image and save it into the path. 

The python script for producing this procedure without using the GUI is given in _blender_utils/output_zpass.py_. The termnial command is 
```
$blender_path/blender -b $.blendfile --python output_zpass.py --$python_args 
```
Rendering the depth pass is very fast in Blender. You can do more interesting things with blender's python library `bpy`, as illustrated in the _lighting\_multiple\_output.py_.

2. Render transient images. This is a one-line command
```
$pbrt_path/pbrt $pbrt_file
```
which will produce by default 256 transient images in `$pwd`. Typically it taks 3~5 minutes to render on a multicore CPU. A batch processing file is given in _/pbrt-v3-tof/example/batch_run_pbrt.sh_.

3. Process the transient images. Here we use MatLab for this purpose. The code is provided in _/transient\_processing/example/transient_to_depth.m_. 

It is important that the _.blend_ file and _.pbrt_ file correspond to the same scene at the same camera viewpoint. There are differences between the coodinate system used by ___pbrt-v3-tof___ and __Blender__.  For example, if you want to produce a _.pbrt_ file from a _.blend_ file, first use Blender's "Export as _.obj_" function and choose the option "-Z forward & Y up", and then further transform it into _.pbrt_ using __obj2pbrt__ provided in the __pbrt-v3-tof__ package.  You often have to make sure the rendering result are consistent. If they are still not consistent, you may also want to check other parameters such as fov, image resolution, etc. in both Blender's setting and the setting in the _.pbrt_ file. As far as I know, blender refer to the horizontal dimension for FOV while pbrt uses the vertical dimension.

### Material augmentation in .pbrt file

In _/pbrt_material_augmentation_ we have some utility functions on augmenting material properties in .pbrt file using MatLab. What it does is to replace the material parameters with some prescribed or random numbers. You should refer to the format of .pbrt files in <https://www.pbrt.org/fileformat-v3.html>. Note in particular that a _.pbrt_ file can refer to other _.pbrt_ files. This will come handy if we have a material library _.pbrt_ file as we do assume here.

### Additional Resources

* This pipeline is heavily influenced by Su et al.'s repo: <https://vccimaging.org/Publications/Su2018EndToEndTOF/>. You can find similar working examples and scene files there.
* The principle of using transient rendering to approximate multi-path interference error in time-of-flight image is described in many places. See e.g. 
    * [Su et al.'s Paper](https://vccimaging.org/Publications/Su2018EndToEndTOF/Su2018EndToEndTOF.pdf) and  [Guo et al.'s Paper](https://research.nvidia.com/sites/default/files/pubs/2018-09_Tackling-3D-ToF/tof_eccv18_0.pdf)

* I have a exposition about the principle of path tracing in [here](https://sylqiu.blogspot.com/2019/06/notes-on-light-transport-in-graphics.html). 
Transient rendering is briefly touched. For full reference, please go to <https://www.pbrt.org/>, one of the best expositions available online.
* Blender tutorial: <https://www.youtube.com/user/AndrewPPrice>

-------------------
## calib.bin files
In the Drive folder there is a calib.bin file, which should be used if mvg_aug is set to be true when training. There is another calib.bin file in the test_real folder, which contains a sample real test image. The calib.bin file there is for real calibration.

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

## Future Work

* __Can we possibly find the "correct material"?__ In this pipeline, given fixed scene geometry, multi-path interference error is determined by the material properties. Determining what meterial specification should coincide with the sensor's capture is not done here. 
* __Sensor error simulation__ In Guo's FLAT dataset, they implemented Kinect's camera function and modelled its noise distribution. Here we do not have sensor specific data pipeline.
* __GPU acceleration__ PBRT currently only parallelizes on CPU.
* __PBRT-Blender materal correspondence__ This is specifically important if one wants to use RGB images from Blender. 

Disclaimer: This software and related data are published for academic and non-commercial use only.

