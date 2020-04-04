# Periocular in the wild: Dual-stream CNN and OC-LBCP


### Introduction
This work contributes a novel descriptor OC-LBCP and Dual-stream CNN for periocular in the wild.

We also provide the codes as follows:
  1) OC-LBCP (see [OC-LBCP](https://github.com/tiongleslie/DSCNN_OCLBCP/tree/master/OC-LBCP))
  2) Pre-trained model of Dual-stream CNN for periocular in the wild (see [DSCNN](https://github.com/tiongleslie/DSCNN_OCLBCP/tree/master/DSCNN))



### Pre-trained Model
Please download the pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1ktAxZYLMIRLjVQw6h89Er42XjfLvtRJ6?usp=sharing) and ensure that you set up our directories as follows:
```
DSCNN_OCLBCP
├── model_result
│   ├── mat
│   └── save
└── test_data
    ├── descriptor
    └── rgb
```



### Compatibility
We tested the codes with:

##### DSCNN
  1) Tensorflow-GPU 1.13.1 under Ubuntu OS 18.04 LTS and Anaconda3 (Python 3.7)
  2) Tensorflow-GPU 1.13.1/Tensorflow 1.13.1 under Windows 10 and Anaconda3 (Python 3.7)
  
##### OCLBCP
  1) MATLAB 2018b and MATLAB 2019b



### Requirements
  1) [Anaconda3](https://www.anaconda.com/distribution/#download-section)
  2) [TensorFlow-GPU 1.13.1 or Tensorflow 1.13.1](https://www.tensorflow.org/install/pip)
  3) [PIL](https://anaconda.org/anaconda/pillow)
  4) [SciPy](https://anaconda.org/anaconda/scipy)
  5) [Matplotlib](https://anaconda.org/conda-forge/matplotlib)
  6) [Numpy 1.16.4](https://pypi.org/project/numpy/1.16.4/)
  7) [MATLAB](https://uk.mathworks.com/products/matlab.html)



### Dataset
Please refer to our paper [1] for the dataset information.



### Citation
Please cite us if you are using our model or dataset in your research work: <br />


  [1] Leslie Ching Ow Tiong, Andrew Beng Jin Teoh and Yunli Lee, “Periocular Recognition in the Wild: Implementation of RGB-OCLBCP Dual-Stream CNN”, *Appl. Sci.*, 2019, 9(13), (see [link](https://doi.org/10.3390/app9132709)).
