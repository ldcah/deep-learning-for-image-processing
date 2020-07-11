import cv2 as cv
from torchvision import transforms
import torch
from PIL import Image

data_transform = transforms.Compose([transforms.ToTensor()])
cvNet = cv.dnn.readNetFromONNX('..\weights_res.pth.onnx')
imgpath = r'E:\pytorch_refuse\2.bmp'
img = cv.imread(imgpath)

# opencv 图像归一化
blob = cv.dnn.blobFromImage(img, swapRB=True, scalefactor=1 / 255)
cvNet.setInput(blob)
cvOut = cvNet.forward()
print(cvOut)

img2 = Image.open(imgpath)
# plt.imshow(img)
# [N, C, H, W]
img_d = data_transform(img2)
img_dexpand = torch.unsqueeze(img_d, dim=0)
img_nup = img_dexpand.numpy()
cvNet.setInput(img_nup)
cvOut2 = cvNet.forward()
print(cvOut2)

tensorOut = torch.from_numpy(cvOut2[0])
pre = torch.softmax(tensorOut,dim=0)
print(pre)

cv.imshow('img', img)
cv.waitKey()
