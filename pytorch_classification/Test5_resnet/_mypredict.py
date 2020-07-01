import sys

from PyQt5.QtWidgets import QApplication

from ui.ui_mainwindow_ext import Ui_MainWindow_Ext

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Ui_MainWindow_Ext()
    mainWindow.show()
    sys.exit(app.exec_())


# .spec
# import sys
# sys.setrecursionlimit(1000000)
#
# datas_a=[(r'D:\Pro\Anaconda3\Lib\site-packages\PyQt5\sip.cp36-win_amd64.pyd','PyQt5')]