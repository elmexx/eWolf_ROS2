from torch.autograd import Variable

import torch.onnx
import torchvision
import torch
import numpy as np
from unet import UNet

model = UNet(n_channels=3, n_classes=1)

inputnp = np.zeros((256,512,3)).astype(np.double)

inputimg = np.expand_dims(np.rollaxis(inputnp,axis=2,start=0),axis=0)

dummy_input = Variable(torch.from_numpy(inputimg))
state_dict = torch.load('./model/CP_epoch8.pth')
model.load_state_dict(state_dict)
torch.onnx.export(model, dummy_input, "CP_epoch8.onnx")
