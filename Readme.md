# Pipeline Associated Learning
Inspired by [Associated Learning](https://github.com/Hibb-bb/AL) ([ICLR 2022](https://in.ncu.edu.tw/~hhchen/academic_works/wu22-associated.pdf)), we propose Pipeline AL and do lots of experiments within performance analysis tools to complement the performance of AL on parallel training. Our method shows that AL can relieve the backward-locking issue via pipieline and speed up each training epoch.

## Environments
We use image **pytorch-22.08-py3** from TWCC refer to [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-08.html#rel_22-08), the version of some main packages are below:
- Python 3.8.13
- CUDA 11.7.1
- Pytorch 1.13

## Datasets
For CV dataset, we provide cifar10/cifar100/tinyImageNet refer to [Supervised Contrastive Parallel Learning](https://github.com/ChengKai-Wang/Supervised-Contrastive-Parallel-Learning), download [tiny-imagenet-200.zip](https://drive.google.com/file/d/1R5QMeXAL_8XYqaDiGFFFoM1IwiJ5ZcBJ/view?usp=sharing) and unzip it directly if you want to run this dataset.

For NLP dataset, we provide datasets refer to [Associated Learning](https://github.com/Hibb-bb/AL), download [IMDB_Dataset.csv](https://drive.google.com/file/d/1GRyOQs6TT0IXKDyha6zNjinmvREKyeuV/view?usp=sharing) to the root of repo if you want to run this dataset.

In theory, you can run our code with dataset mentioned above, but we only test the following dataset:
- tinyImageNet
- AGNews
- Dbpedia
- IMDb

## Embeddings
Please download Glove [here](https://drive.google.com/file/d/17UaPMnhIjXaDLZAfOUaZ79Ev6NRhRISA/view?usp=sharing) and unzip it directly.

## Execution
For CV task, please run:
```
python cvmain.py <dataset> <model> -r <repeat times>
```

For NLP task, please run:
```
python nlpmain.py <dataset> <model> -r <repeat times>
```

To set hyperparameters, please modify **cvconfig.json** or **nlpconfig.json**, and make sure the length of bp_gpu_list / al_gpu_list are equal to number of model layers:

| Model        | BP     | AL      |
|--------------|--------|---------|
| VGG          | 5      | 4       |
| Resnet       | 6      | 5       |
| LSTM         | 5      | 4       |
| Transformer  | 5      | 4       |

Other settings can find in our thesis.

## Pipeline AL with your own BP model
We provide ALPackage for you to simply transfer layers of BP model into layers of AL model, you can simply init an AL layer with your function f, g, b, h, or override behavior of ENC, AE and AL class if needed.

For more information, please refer to our thesis, or take a look at model in ALPackage.
