<img src="C:\Users\xdluo\AppData\Roaming\Typora\typora-user-images\image-20220427230733905.png" alt="image-20220427230733905" style="zoom:20%;" />

<h1 align="center">
  KnowER
</h1>

<p align="center">
    <b>muKG:</b> <b></b>  muKG is a library for multi-source <b>knowledge
        graph embedding and reasoning</b>. It supports multiple deep learning libraries (PyTorch and TensorFlow 2), multiple embedding tasks (link prediction, entity alignment, entity typing, and multi-source link prediction. 
</p>



## Introduction of muKG

### Overview 

We use  [Python](https://www.python.org/) ,  [Tensorflow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) to develop the basic framework of **KnowER**.  And using [RAY](https://www.ray.io/) for distributed training. The software architecture is illustrated in the following Figure. 

![image-20220507103409697](C:\Users\xdluo\AppData\Roaming\Typora\typora-user-images\image-20220507103409697.png)



Compared with other existing KG systems, KnowER has the following competitive features.

👍**Comprehensive.** KnowER is a full-featured Python library for representation learning over a single KG or multi-source KGs. 
  It is compatible with the two widely-used deep learning libraries [PyTorch](https://pytorch.org/) and [TensorFlow 2](https://www.tensorflow.org/), and can therefore be easily integrated into downstream applications. It integrates a variety of KG embedding models and supports four KG tasks including link prediction, entity alignment, entity typing, and multi-source link prediction.

⚡**Fast and scalable.** KnowER provides advanced implementations of KG embedding techniques with the support of multi-process and multi-GPU parallel computing, making it fast and scalable to large KGs.

🤳**Easy-to-use.** KnowER provides simplified pipelines of KG embedding tasks for easy use. Users can interact with KnowER with both method APIs and the command line. It also has high-quality documentation.

😀**Continuously updated.** Our team will keep up-to-date on new related techniques and integrate new (multi-source) KG embedding models, tasks, and datasets into KnowER. We will also keep improving existing implementations.

  

### 	Package Description

```
muKG/
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



## Getting Started 🚀

### Dependencies![python3](https://img.shields.io/badge/Python3-green.svg?style=flat-square)

muKG supports PyTorch and TensorFlow 2 deep learning libraries, users can choose one of the following two dependencies according to their preferences.

* Torch 1.10.2  |  Tensorflow 2.x 
* Ray 1.12.0    
* Scipy
* Numpy
* Igraph 
* Pandas
* Scikit-learn
* Gensim
* Tqdm

### Installation 🔧

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
### Usage 📝

Here we provide tutorials of using command line as well as editing main.py file. The following is an example about how to use muKG in Python.

```python
model_name = 'model name'
kg_task = 'selected KG task'
if kg_task == 'ea':
	args = load_args("hyperparameter file folder of entity alignment task")
elif kg_task == 'lp':
	args = load_args("hyperparameter file folder of link prediction task")
else:
	args = load_args("hyperparameter file folder of entity typing task")
kgs = read_kgs_from_folder()
if kg_task == 'ea':
	model = ea_models(args, kgs)
elif kg_task == 'lp':
	model = kge_models(args, kgs)
else:
	model = et_models(args, kgs)
model.get_model('TransE')
model.run()
model.test()
```

You can run a model on a dataset with the following command line:

```
python main_args.py -t lp -m transe -o train -d data/FB15K
```

You can also use the following command line to train your model with multi-GPU and multi-processing.

```bash
python main_args.py -t lp -m transe -o train -d data/FB15K -r gpu:2 -w 2  
```



## Models hub 🏠

muKG has implemented 26 KG models. The citation for each models corresponds to either the paper
describing the model. It is available for you to add your own model to muKG.

### KGE models

| Name     | Citation                                                                                                                |
| -------- |-------------------------------------------------------------------------------------------------------------------------|
| TransE   | [Bordes *et al.*, 2013](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) |
| TransR   | [Lin *et al.*, 2015](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523/)                           |
| TransD   | [Ji *et al.*, 2015](http://www.aclweb.org/anthology/P15-1067)                                                           |
| TransH   | [Wang *et al.*, 2014](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546)                          |
| TuckER   | [Balažević *et al.*, 2019](https://arxiv.org/abs/1901.09590)                                                            |
| RotatE   | [Sun *et al.*, 2019](https://arxiv.org/abs/1902.10197v1)                                                                |
| SimplE   | [Kazemi *et al.*, 2018](https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs)     |
| RESCAL   | [Nickel *et al.*, 2011](http://www.cip.ifi.lmu.de/~nickel/data/paper-icml2011.pdf)                                      |
| ComplEx  | [Trouillon *et al.*, 2016](https://arxiv.org/abs/1606.06357)                                                            |
| Analogy  | [Liu *et al.*, 2017](https://arxiv.org/abs/1705.02426)                                                                  |
| DistMult | [Yang *et al.*, 2014](https://arxiv.org/abs/1412.6575)                                                                  |
| HolE     | [Nickel *et al.*, 2016](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828)                      |
| ConvE    | [Dettmers *et al.*, 2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17366)                              |

### EA models

| Name      | Citation                                                                                                    |
| --------- |-------------------------------------------------------------------------------------------------------------|
| MTransE   | [Chen *et al.*, 2017](https://www.ijcai.org/proceedings/2017/0209.pdf)                                      |
| IPTransE  | [Zhu *et al.*, 2017 ](https://www.ijcai.org/proceedings/2017/0595.pdf)                                      |
| BootEA    | [Sun *et al.*, 2018](https://www.ijcai.org/proceedings/2018/0611.pdf)                                       |
| JAPE      | [Sun *et al.*, 2017](https://link.springer.com/chapter/10.1007/978-3-319-68288-4_37)                        |
| IMUSE     | [He *et al.*, 2019](https://link.springer.com/content/pdf/10.1007%2F978-3-030-18576-3_22.pdf)               |
| RDGCN     | [Wu *et al.*, 2019](https://www.ijcai.org/proceedings/2019/0733.pdf)                                        |
| AttrE     | [Trisedya *et al.*, 2019](https://people.eng.unimelb.edu.au/jianzhongq/papers/AAAI2019_EntityAlignment.pdf) |
| SEA       | [Pei *et al.*, 2019](https://dl.acm.org/citation.cfm?id=3313646)                                            |
| GCN-Align | [Wang *et al.*, 2018](https://www.aclweb.org/anthology/D18-1032)                                            |
| RSN4EA    | [Guo *et al.*, 2019](http://proceedings.mlr.press/v97/guo19c/guo19c.pdf)                                    |

### ET models

| Name   | Citation                                                     |
| ------ | ------------------------------------------------------------ |
| TransE | [Bordes *et al.*, 2013](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) |
| RESCAL | [Nickel *et al.*, 2011](http://www.cip.ifi.lmu.de/~nickel/data/paper-icml2011.pdf) |
| HolE   | [Nickel *et al.*, 2016](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828) |

## Datasets hub 🏠

muKG has bulit in 16 KG datasets for different downstream tasks. Here we list the number of entities, relations, train triples, valid triples and test triples for these datasets. You can prepare your own datasets in the Datasets hub.

### KGE datasets

| Datasets Name | Entities | Relations | Train   | Valid | Test    | Citation                                                     |
| ------------- | -------- | --------- | ------- | ----- | ------- | ------------------------------------------------------------ |
| FB15K         | 14951    | 1345      | 483142  | 50000 | 59071   | [Bordes *et al*., 2013](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) |
| FB15K237      | 14541    | 237       | 272115  | 17535 | 20466   | [Bordes *et al*., 2013](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) |
| WN18RR        | 40943    | 11        | 86835   | 3034  | 3134    | [Toutanova *et al*., 2015](https://www.aclweb.org/anthology/W15-4007/) |
| WN18          | 40943    | 18        | 141442  | 5000  | 5000    | [Bordes *et al*., 2014](https://arxiv.org/abs/1301.3485)     |
| WN11          | 38588    | 11        | 112581  | 2609  | 10544   | dbPEDIA                                                      |
| DBpedia50     | 49900    | 654       | 23288   | 399   | 10969   | [Shi *et al*., 2017](https://arxiv.org/abs/1711.03438)       |
| DBpedia500    | 517475   | 654       | 3102677 | 10000 | 1155937 |                                                              |
| Countries     | 271      | 2         | 1111    | 24    | 24      | [Bouchard *et al*., 2015](https://www.aaai.org/ocs/index.php/SSS/SSS15/paper/view/10257/10026) |
| FB13          | 75043    | 13        | 316232  | 5908  | 23733   |                                                              |
| Kinsip        | 104      | 25        | 8544    | 1086  | 1074    | [Kemp *et al*., 2006](https://www.aaai.org/Papers/AAAI/2006/AAAI06-061.pdf) |
| Nations       | 14       | 55        | 1592    | 199   | 201     | [`ZhenfengLei/KGDatasets`](https://github.com/ZhenfengLei/KGDatasets) |
| NELL-995      | 75492    | 200       | 149678  | 543   | 3992    |                                                              |
| UMLS          | 75492    | 135       | 5216    | 652   | 661     | [`ZhenfengLei/KGDatasets`](https://github.com/ZhenfengLei/KGDatasets) |

### EA datasets

| Datasets name    | Entities | Relations | Triples | Citation                                                     |
| ---------------- | -------- | --------- | ------- | ------------------------------------------------------------ |
| OpenEA supported | 15000    | 248       | 38265   | [Sun *et al*., 2020](http://www.vldb.org/pvldb/vol13/p2326-sun.pdf) |

### ET datasets

| Datasets name | Entities | Relations | Triples | Types | Citation                                                     |
| ------------- | -------- | --------- | ------- | ----- | ------------------------------------------------------------ |
| FB15K-ET      | 15000    | 248       | 38265   |       | [Moon *et al*., 2017]([Learning Entity Type Embeddings for Knowledge Graph Completion (persagen.com)](https://persagen.com/files/misc/Moon2017Learning.pdf)) |



## Utils 📂

### Sampler

**Negative sampler:**

muKG includes several negative sampling methods to randomly generate negative examples.

- **Uniform negative sampling**
- **Self-adversarial negative sampling** 
- **Truncated negative sampling** 

**Path sampler:**
The Path sampler is to support some embedding models that are built by modeling the paths of KGs, such as IPTransE and RSN4EA. It can generate relational path like (e_1, r_1, e_2, r_2, e_3), entity path like (e_1, e_2, e_3), and relation path like (r_1, r_2).

**Subgraph sampler:**

The subgraph sampler is to support GNN-based embedding models like GCN-Align and AliNet. It can generate both first-order (i.e., one-hop) and high-order (i.e., multi-hop) neighborhood subgraphs of entities.

### Multi-GPU and multi-processing computation

We use [Ray](https://www.ray.io/) to provide a uniform and easy-to-use interface for multi-GPU and multi-processing computation. The following figure shows our Ray-based implementation for parallel computing and the code snippet to use it. Users can set the number of CPUs or GPUs used for model training.

![image-20220507172436866](C:\Users\xdluo\Desktop\Knowsys\resources\image-20220507172436866.png)

You can use the following command line to train your model with multi-GPU and multi-processing.

```bash
python main_args.py -t lp -m transe -o train -d data/FB15K -r gpu:2 -w 2  
```






