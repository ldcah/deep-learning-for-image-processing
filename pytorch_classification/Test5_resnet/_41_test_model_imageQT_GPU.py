import os
import sys
from configparser import ConfigParser

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import *
import torch
from model import resnet34
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

class Picture(QWidget):
    def __init__(self):
        super(Picture, self).__init__()

        self.resize(800, 650)
        self.setWindowTitle("测试图像")

        self.label4image = QLabel(self)
        self.label4image.setText("显示图片")
        self.label4image.setFixedSize(672, 512)
        self.label4image.move(10, 20)

        self.label4image.setStyleSheet("QLabel{background:white;}"
                                       "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                       )
        btnOpenImage = QPushButton(self)
        btnOpenImage.setText("打开图片")
        btnOpenImage.move(700, 60)
        btnOpenImage.clicked.connect(self.openimage)

        self.btnLoadModel = QPushButton(self)
        self.btnLoadModel.setText("读取模型")
        self.btnLoadModel.move(700, 20)
        self.btnLoadModel.clicked.connect(self.readmodel)

        self.btnLoadModel2 = QPushButton(self)
        self.btnLoadModel2.setText("批量图片")
        self.btnLoadModel2.move(700, 100)
        self.btnLoadModel2.clicked.connect(self.openImages)

        self.lbl4ModelName = QLabel(self)
        self.lbl4ModelName.setText("权重文件")
        self.lbl4ModelName.setFixedSize(700, 30)
        self.lbl4ModelName.move(10, 540)

        self.lbl4ImageName = QLabel(self)
        self.lbl4ImageName.setText("图像名称")
        self.lbl4ImageName.setFixedSize(700, 30)
        self.lbl4ImageName.move(10, 570)

        self.lbl4result = QLabel(self)
        self.lbl4result.setText("显示结果")
        self.lbl4result.setFixedSize(700, 30)
        self.lbl4result.move(10, 600)

        self.my_model_pre = None

    def readmodel(self):
        try:
            modelName, imgType = QFileDialog.getOpenFileName(self, "打开权重文件 pth", "", "*.pth")
            if modelName != "":
                self.lbl4ModelName.setText("模型加载中.....")
                self.my_model_pre = mymodel(modelName, modelName + ".json")
                self.lbl4ModelName.setText("模型加载完成。" + modelName)
            else:
                self.lbl4ModelName.setText("未加载权重数据！")
        except Exception as exc:
            print("{0}{1}".format(type(exc).__name__, str(exc)))
            self.lbl4ModelName.setText("加载模型异常：" + modelName)

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.bmp;;*.png;;All Files(*)")

        if imgName != "":
            # 显示图片名称
            self.lbl4ImageName.setText(imgName)
            jpg = QtGui.QPixmap(imgName).scaled(self.label4image.width(), self.label4image.height())
            self.label4image.setPixmap(jpg)
            reslt = ''
            if (self.my_model_pre != None):
                try:
                    reslt = self.my_model_pre.predict(imgName)
                except Exception:
                    reslt = '图片无法预测'
                self.lbl4result.setText(reslt)

    def openImages(self):
        """
        批量测试图片
        :return:
        """
        file_path = QFileDialog.getExistingDirectory(self, "选择存储文件夹")
        if file_path != "":
            if (self.my_model_pre != None):
                # 获取目录下的图片文件
                with open(file_path + "\\00data.csv", mode="w")as f:
                    for imagename in os.listdir(file_path):
                        if imagename.endswith(".bmp"):
                            reslt = self.my_model_pre.predict(file_path + "\\" + imagename)
                            print(reslt)
                            self.lbl4result.setText(reslt)
                            f.writelines(file_path + "/" + imagename + "," + imagename + "," + reslt + "\r")
                            jpg = QtGui.QPixmap(file_path + "/" + imagename).scaled(self.label4image.width(),
                                                                                    self.label4image.height())
                            self.label4image.setPixmap(jpg)


class mymodel():

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


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    my = Picture()
    my.show()
    sys.exit(app.exec_())
