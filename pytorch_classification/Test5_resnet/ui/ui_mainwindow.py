# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1051, 758)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 0, 1021, 541))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.groupBox = QtWidgets.QGroupBox(self.horizontalLayoutWidget)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.lblShowImage = QtWidgets.QLabel(self.groupBox)
        self.lblShowImage.setGeometry(QtCore.QRect(10, 10, 672, 512))
        self.lblShowImage.setObjectName("lblShowImage")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(700, 0, 311, 531))
        self.groupBox_2.setObjectName("groupBox_2")
        self.btnOpenImage = QtWidgets.QPushButton(self.groupBox_2)
        self.btnOpenImage.setGeometry(QtCore.QRect(10, 20, 75, 41))
        self.btnOpenImage.setObjectName("btnOpenImage")
        self.btnOpenImageS = QtWidgets.QPushButton(self.groupBox_2)
        self.btnOpenImageS.setGeometry(QtCore.QRect(220, 20, 75, 41))
        self.btnOpenImageS.setObjectName("btnOpenImageS")
        self.horizontalLayout_2.addWidget(self.groupBox)
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 0, 2, 2))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 559, 1021, 151))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.lbl4ModelName = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.lbl4ModelName.setObjectName("lbl4ModelName")
        self.verticalLayout.addWidget(self.lbl4ModelName)
        self.lbl4ImageName = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.lbl4ImageName.setObjectName("lbl4ImageName")
        self.verticalLayout.addWidget(self.lbl4ImageName)
        self.lbl4result = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.lbl4result.setObjectName("lbl4result")
        self.verticalLayout.addWidget(self.lbl4result)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1051, 23))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoadModel = QtWidgets.QAction(MainWindow)
        self.actionLoadModel.setObjectName("actionLoadModel")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.actionLoadModel)
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "预测图像"))
        self.lblShowImage.setText(_translate("MainWindow", "显示图片"))
        self.groupBox_2.setTitle(_translate("MainWindow", "操作"))
        self.btnOpenImage.setText(_translate("MainWindow", "读取图片"))
        self.btnOpenImageS.setText(_translate("MainWindow", "图片目录"))
        self.lbl4ModelName.setText(_translate("MainWindow", "模型文件"))
        self.lbl4ImageName.setText(_translate("MainWindow", "图片信息"))
        self.lbl4result.setText(_translate("MainWindow", "结果信息"))
        self.menuFile.setTitle(_translate("MainWindow", "文件"))
        self.actionLoadModel.setText(_translate("MainWindow", "加载"))
        self.actionExit.setText(_translate("MainWindow", "退出"))