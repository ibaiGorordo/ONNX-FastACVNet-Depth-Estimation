# ONNX-FastACVNet-Stereo-Depth-Estimation
 Python scripts performing stereo depth estimation using the Fast-ACVNet model in ONNX.
 
![!Fast-ACVNet detph estimation](https://github.com/ibaiGorordo/ONNX-FastACVNet-Depth-Estimation/blob/main/doc/img/out.jpg)
*Stereo depth estimation on the cones images from the Middlebury dataset (https://vision.middlebury.edu/stereo/data/scenes2003/)*

# Requirements

 * Check the **requirements.txt** file.
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.

# Installation
```
git clone https://github.com/ibaiGorordo/ONNX-FastACVNet-Depth-Estimation.git
cd ONNX-FastACVNet-Depth-Estimation
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

# ONNX model
The models were converted from the Pytorch implementation below by [PINTO0309](https://github.com/PINTO0309), download the models from the download script in [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/338_Fast-ACVNet) and save them into the **[models](hhttps://github.com/ibaiGorordo/ONNX-FastACVNet-Depth-Estimation/tree/main/models)** folder. 
- The License of the models is MIT License: https://github.com/gangweiX/Fast-ACVNet/blob/main/LICENSE.md

# Pytorch model
The original Pytorch model can be found in this repository: https://github.com/gangweiX/Fast-ACVNet
 
# Examples

 * **Image inference**:
 ```
 python image_depth_estimation.py
 ```

 * **Video inference**:
 ```
 python video_depth_estimation.py
 ```
 ![!Fast-ACVNet depth estimation](https://github.com/ibaiGorordo/ONNX-FastACVNet-Depth-Estimation/blob/main/doc/img/fastacvnet_malaga_urban.gif)
 
 *Original video: Málaga Stereo and Laser Urban dataset, reference below*

 * **Driving Stereo dataset inference**: https://youtu.be/az4Z3dp72Zw
 ```
 python driving_stereo_test.py
 ```
 ![!CREStereo depth estimation](https://github.com/ibaiGorordo/ONNX-FastACVNet-Depth-Estimation/blob/main/doc/img/fastacvnet_driving_stereo.gif)
  
 *Original video: Driving stereo dataset, reference below*
  
# References:
* Fast-ACVNet model: https://github.com/gangweiX/Fast-ACVNet
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* Driving Stereo dataset: https://drivingstereo-dataset.github.io/
* Málaga Stereo and Laser Urban dataset: https://www.mrpt.org/MalagaUrbanDataset
* Original paper: https://arxiv.org/abs/2209.12699
