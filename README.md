# ImageClassificationTemplate
This is an image classification finetune template using torchvision or timm.

### 1.prerequisites

python3.7+、gpu

### 2.installation

`pip install -r requirements.txt`

### 3.Support List

- **Multi-GPU Training Solution**
  + [x] [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/nn.html#distributeddataparallel) [Built-in function of Pytorch]
- **Pretrain Model**
  + [x] [totchvision](https://pytorch.org/vision/stable/index.html)
  + [x] [timm](https://github.com/rwightman/pytorch-image-models)
  + [x] local
- **Deploy**
  + [x] onnx(support simplify and optimize)
  + [x] mnn

- **Log**
  + [ ] logger
  + [ ] tensorboard
