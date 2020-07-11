import torch
from torch.autograd import Variable

from model import resnet34

print('Torch Version', torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

example = torch.rand(1, 3, 512, 672).to(device)

model = resnet34(5).to(device)
checkpoint = torch.load("..\weights_res.pth", map_location=lambda storage, loc: storage.cuda(0))
model.load_state_dict(checkpoint)

model.eval()
torch.onnx.export(model, example, "..\weights_res.pth.onnx", verbose=True)

print("Translate end")
