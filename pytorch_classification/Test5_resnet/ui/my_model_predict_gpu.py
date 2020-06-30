import json
import time
import torch
from PIL import Image
from torchvision import transforms
from model import resnet34, resnet50


class MyModel4Prdict():

    def __init__(self, model_weight_path, json_file):
        # self.data_transform = transforms.Compose([transforms.Resize(256),
        #                                           transforms.CenterCrop(224),
        #                                           transforms.ToTensor(),
        #                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        #
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.device_cpu = torch.device("cpu")
        self.data_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        json_filef = open(json_file, "r", encoding="utf-8")

        self.class_indict = json.load(json_filef)

        # create model
        if (str.find(model_weight_path, "res50.pth") > 0):
            self.model = resnet50(num_classes=len(self.class_indict))  # res50
        else:
            self.model = resnet34(num_classes=len(self.class_indict))
        # load model weights
        self.model.load_state_dict(torch.load(model_weight_path))
        self.model.eval()
        self.model.to(self.device)

    def predict(self, imgpath):
        time_start = time.time()
        # load image
        img = Image.open(imgpath)
        # plt.imshow(img)
        # [N, C, H, W]
        img = self.data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # output = torch.squeeze(self.model(img))
            output = torch.squeeze(self.model(img.to(self.device)))
            predict = torch.softmax(output, dim=0).to(self.device_cpu)
            predict_cla = torch.argmax(predict).numpy()

        # percent = str(predict[predict_cla].numpy())
        percent = "{:.4f}".format(predict[predict_cla].numpy())
        res_name = self.class_indict[str(predict_cla)]
        # 耗时
        use_time = "{:.3f}".format(time.time() - time_start)
        res_joson = "{0},{1},{2},{3}s".format(predict_cla, percent, res_name, use_time)
        return res_joson
