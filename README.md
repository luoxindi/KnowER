<img src="C:\Users\xdluo\AppData\Roaming\Typora\typora-user-images\image-20220427230733905.png" alt="image-20220427230733905" style="zoom:20%;" />

<h1 align="center">
  KnowER
</h1>

<p align="center">
    <b>KnowER:</b> <b></b></b>  KnowER is a library for multi-source <b>knowledge
        graph embedding and reasoning</b>. It supports multiple deep learning libraries (PyTorch and TensorFlow 2), multiple embedding tasks (link prediction, entity alignment, entity typing, and multi-source link prediction. 
</p>



## Introduction of KnowER

### Overview 

We use  [Python](https://www.python.org/) ,  [Tensorflow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) to develop the basic framework of **KnowER**.  And using [RAY](https://www.ray.io/) for distributed training. The software architecture is illustrated in the following Figure. 

![image-20220507103409697](C:\Users\xdluo\AppData\Roaming\Typora\typora-user-images\image-20220507103409697.png)



Compared with other existing KG systems, KnowER has the following competitive features.

- **Comprehensive.** KnowER is a full-featured Python library for representation learning over a single KG or multi-source KGs. 
  It is compatible with the two widely-used deep learning libraries [PyTorch](https://pytorch.org/) and [TensorFlow 2](https://www.tensorflow.org/), and can therefore be easily integrated into downstream applications. It integrates a variety of KG embedding models and supports four KG tasks including link prediction, entity alignment, entity typing, and multi-source link prediction.

- **Fast and scalable.** KnowER provides advanced implementations of KG embedding techniques with the support of multi-process and multi-GPU parallel computing, making it fast and scalable to large KGs.

- **Easy-to-use.** KnowER provides simplified pipelines of KG embedding tasks for easy use. Users can interact with KnowER with both method APIs and the command line. It also has high-quality documentation.

- **Continuously updated.** Our team will keep up-to-date on new related techniques and integrate new (multi-source) KG embedding models, tasks, and datasets into KnowER. We will also keep improving existing implementations.

  

### 	Package Description

```
KnowER/
├── src/
│   ├── py/: a Python-based toolkit used for the upper layer of KnowER
		|── data/: a collection of datasets used for knowledge graph reasoning
		|── args/: json files used for configuring hyperparameters of training process
		|── evaluation/: package of the implementations for supported downstream tasks
		|── load/: toolkit used for data loading and processing
		|── base/: package of the implementations for different initializers, losses and optimizers
		|── util/: package of the implementations for checking virtual environment
│   ├── tf/: package of the implementations for KGE models, EA models and ET models in TensorFlow 2
│   ├── torch/: package of the implementations for KGE models, EA models and ET models in PyTorch
```



## Getting Started

### Dependencies![python3](https://img.shields.io/badge/Python3-green.svg?style=flat-square)

KnowER supports PyTorch and TensorFlow 2 deep learning libraries, users can choose one of the following two dependencies according to their preferences.

* Torch 1.10.2  |  Tensorflow 2.x 
* Ray 1.12.0    
* Scipy
* Numpy
* Igraph 
* Pandas
* Scikit-learn
* Gensim
* Tqdm

### Installation

We suggest you create a new conda environment firstly.  We provide two installation instructions for tensorflow-gpu (tested on 1.2.1) and pytorch (tested on 1.10.2). Note that there is a difference between the Ray 1.10.0 and Ray 1.12.0 in batch generation module. The Ray 1.10.0 is used as an example.

TensorFlow 2  

```bash
conda create -n knower python=3.8
conda activate knower
conda install tensorflow-gpu==2.3.0
conda install -c conda-forge python-igraph
pip install -U ray==1.12.0
```

To install PyTorch, you must install [Anaconda](https://www.anaconda.com/) and follow the instructions on the PyTorch website. For example, if you’re using CUDA version 11.3, use the following command:

PyTorch

```bash
conda create -n knower python=3.8
conda activate knower
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge python-igraph
pip install -U ray==1.12.0
```

The latest code can be installed by the following instructions:

```bash
git clone https://github.com/luoxindi/KnowER.git KnowER
cd KnowER
pip install -e .
```





