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
  + [x] totchvision
  + [x] timm
- **Deploy**
  + [x] onnx
  + [x] mnn
