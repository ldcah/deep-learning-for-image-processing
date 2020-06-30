import sys

from PyQt5.QtWidgets import QApplication

from ui.ui_mainwindow_ext import Ui_MainWindow_Ext

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Ui_MainWindow_Ext()
    mainWindow.show()
    sys.exit(app.exec_())
