# Pipeline Associated Learning
Associated learning ([ICLR 2022](https://in.ncu.edu.tw/~hhchen/academic_works/wu22-associated.pdf)) ([Code](https://github.com/Hibb-bb/AL))  can theoretically enhance training efficiency through pipelining.  However, the original proposal lacks the implementation of the pipeline.  This repository implements the pipelined associated learning (pipeline AL) to demonstrate its ability to address the backward-locking issue and increase training throughput.

## Environments
We use image [**pytorch-22.08-py3**](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-08.html#rel_22-08) from [TWCC](https://www.twcc.ai/).  The versions of the main packages are listed below:
- Python 3.8.13
- CUDA 11.7.1
- Pytorch 1.13

## Datasets
We use the following datasets for experiments.
- tinyImageNet
- AGNews
- Dbpedia
- IMDb

If you want to experiment with tinyImageNet, download [tiny-imagenet-200.zip](https://drive.google.com/file/d/1R5QMeXAL_8XYqaDiGFFFoM1IwiJ5ZcBJ/view?usp=sharing) and unzip it.

If you want to experiment with IMDb, download [IMDB_Dataset.csv](https://drive.google.com/file/d/1GRyOQs6TT0IXKDyha6zNjinmvREKyeuV/view?usp=sharing) to the root of the repo.

AGNews and Dbpedia are provided by PyTorch.

## Embeddings
Please download Glove [here](https://drive.google.com/file/d/17UaPMnhIjXaDLZAfOUaZ79Ev6NRhRISA/view?usp=sharing) and unzip it.

## Execution
For CV task (in our case, tinyImageNet), please run:
```
python cvmain.py <dataset> <model> -r <repeat times>
```

For NLP task (in our case, AGNews, Dbpedia, and IMDb), please run:
```
python nlpmain.py <dataset> <model> -r <repeat times>
```

To set up the hyperparameters, please modify **cvconfig.json** or **nlpconfig.json**, and make sure the length of bp_gpu_list / al_gpu_list equals the number of model layers:

| Model        | BP     | AL      |
|--------------|--------|---------|
| VGG          | 5      | 4       |
| Resnet       | 6      | 5       |
| LSTM         | 5      | 4       |
| Transformer  | 5      | 4       |

Other settings can be found in our thesis.

## Pipeline AL with your own BP model
We provide ALPackage to simply the process of transferring from a BP model into its AL form. You can initiate an AL layer with your functions f, g, b, h, or override ENC, AE, and AL classes if needed. The pseudocode below may help you familiarize with the use of the ALPackage.

### Transfer BP model to AL form
Suppose you have a BP model below:
```
class MyBPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = MyBPLayer()
        self.l2 = MyBPLayer()
        self.l3 = MyBPLayer()
        self.l4 = MyBPLayer()

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x
```

You can divide BPModel into several f functions in AL and design your g, b, h function:
```
# For example, we divide the original BPModel into 4 layers and use simple Linear() to init b, g, h function
f = MyBPLayer()
b = Linear()
g = Linear()
h = Linear()

# Import ENC, AE, and AL layers in ALPackage and construct your AL model
from ALPackage.base import *
enc = ENC(f, b)
ae = AE(g, h)

class MyALModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = AL(ENC, AE)
        self.l2 = AL(ENC, AE)
        self.l3 = AL(ENC, AE)
        self.l4 = AL(ENC, AE)

    def forward(self, x, y):
        x, y = self.l1(x, y)
        x, y = self.l2(x, y)
        x, y = self.l3(x, y)
        x, y = self.l4(x, y)
```
### Method in class **AL**
We already defined some methods in the AL layer:
- forward(x, y)         # given x and y, execute forward
- backward()            # execute backward to calculate the local loss and gradient of this AL layer
- update()              # update trainable parameters in this AL layer 
- inference(x, path)    # use for inference, path indicates how to pass x

To train the AL model, call forward/backward/update layer by layer like forward() in MyALModel
For inference, call inference and decide your inference path (f, b, h):
```
# For example, if we want to get the full inference path of MyALModel:
model = MyALModel()
model.l1.inference(x, 'f')
model.l2.inference(x, 'f')
model.l3.inference(x, 'f')
model.l4.inference(x, 'f')
model.l4.inference(x, 'b')
model.l4.inference(x, 'h')
model.l3.inference(x, 'h')
model.l2.inference(x, 'h')
model.l1.inference(x, 'h')
```

### Pipeline your AL (AL-FM)
In our research, we use model parallelism and multithread to pipeline AL and further propose AL-FM. To create the AL-FM model, you need to prepare your AL model and specify each layer to the correct device in forward / inference:
```
model = MyALModel()
model.l1.to('cuda:0')
model.l2.to('cuda:1')
model.l3.to('cuda:2')
model.l4.to('cuda:3')

# forward, and similar idea for inference
x, y = model.l1(x.to('cuda:0'), y.to('cuda:0'))
x, y = model.l2(x.to('cuda:1'), y.to('cuda:1'))
x, y = model.l3(x.to('cuda:2'), y.to('cuda:2'))
x, y = model.l4(x.to('cuda:3'), y.to('cuda:3'))

# or you can use set_device() from utils.py in the repo if you don't override the AL model  
model = MyALModel()
set_device(model, ['cuda:0','cuda:1','cuda:2','cuda:3'])
```

Then use multithread for training:
```
import threading
thread = []
for layer in model:
    x, y = layer(x, y)  # do forward
    thread.append(threading.Thread(target=layer.backward_and_update)) # wrap backward & update to multithread
    thread[-1].start()  # use multithread to complete this

[thread[i].join() for i in range(len(thread))]  # wait until all layers complete

```

For more information, please refer to our thesis, or take a look at the model in ALPackage.
