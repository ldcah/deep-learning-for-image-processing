import os
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from ui.ui_mainwindow import Ui_MainWindow
from ui.my_model_predict_gpu import MyModel4Prdict

# 循环中不刷新界面
gui = QtGui.QGuiApplication.processEvents


class Ui_MainWindow_Ext(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(Ui_MainWindow_Ext, self).__init__(parent)
        self.setupUi(self)
        self.my_model_pre = None  # 推理模型
        self.setSingleSolt()  # 连接 信号与槽

    def setSingleSolt(self):
        """
        设置信号与槽
        :return:
        """
        self.actionLoadModel.triggered.connect(self.slots_actionLoadModel)
        self.actionExit.triggered.connect(self.slots_doExit)
        self.btnOpenImage.clicked.connect(self.openImageClick)
        self.btnOpenImageS.clicked.connect(self.openImageSClick)

    def slots_actionLoadModel(self):
        """
        1.加载模型文件
        :return:
        """
        try:
            modelName, imgType = QFileDialog.getOpenFileName(self, "打开权重文件 pth", "", "*.pth")
            if modelName != "":
                self.lbl4ModelName.setText("模型加载中.....")
                self.my_model_pre = MyModel4Prdict(modelName, modelName + ".json")
                self.lbl4ModelName.setText("模型加载完成。" + modelName)
        except Exception as exc:
            print("{0}{1}".format(type(exc).__name__, str(exc)))
            self.lbl4ModelName.setText("加载模型异常：" + modelName)

    def slots_doExit(self):
        """
        退出
        :return:
        """
        self.centralwidget.parentWidget().close()

    def openImageClick(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.bmp;;*.png;;*.jpg;;All Files(*)")

        if imgName != "":
            # 显示图片名称
            self.lbl4ImageName.setText(imgName)
            jpg = QtGui.QPixmap(imgName).scaled(self.lblShowImage.width(), self.lblShowImage.height())
            self.lblShowImage.setPixmap(jpg)
            reslt = ''
            if (self.my_model_pre != None):
                try:
                    reslt = self.my_model_pre.predict(imgName)
                except Exception as exc:
                    print(str(exc))
                self.lbl4result.setText(reslt)

    def openImageSClick(self):
        """ 批量测试图片
        """
        file_path = QFileDialog.getExistingDirectory(self, "选择存储文件夹")
        if file_path != "":
            if (self.my_model_pre != None):
                # 获取目录下的图片文件
                with open(file_path + "\\00data.csv", mode="w")as f:
                    for imagename in os.listdir(file_path):
                        if imagename.endswith(".bmp"):
                            full_image_name = "{}/{}".format(file_path, imagename)
                            reslt = self.my_model_pre.predict(full_image_name)
                            print(reslt)
                            self.lbl4ImageName.setText(full_image_name)
                            self.lbl4result.setText(reslt)
                            gui()
                            # 保存到文件中
                            f.writelines("{},{},{}\r".format(full_image_name, imagename, reslt))
                            jpg = QtGui.QPixmap(full_image_name).scaled(self.lblShowImage.width(),
                                                                        self.lblShowImage.height())

                            self.lblShowImage.setPixmap(jpg)
                            gui()


# 方便调试用 2020-06-30 ludc
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    from ui.ui_mainwindow_ext import Ui_MainWindow_Ext

    app = QApplication(sys.argv)
    mainWindow = Ui_MainWindow_Ext()
    mainWindow.show()
    sys.exit(app.exec_())
