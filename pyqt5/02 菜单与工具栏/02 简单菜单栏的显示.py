
import sys
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication
from PyQt5.QtGui import QIcon

# 带有菜单File的窗口，该窗口有一个
class Example(QMainWindow):

	def __init__(self):
		super().__init__()

		self.initUI()

	def initUI(self):
		exitAct = QAction(QIcon('exit.png'), '&Exit', self)    # 设置选项动作
		exitAct.setShortcut('Ctrl+Q')               # 设置选项的快捷键
		exitAct.setStatusTip('Exit application')    # 菜单选项的提示信息
		exitAct.triggered.connect(qApp.quit)        # 绑定动作

		self.statusBar()

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(exitAct)

		self.setGeometry(300, 300, 300, 200)
		self.setWindowTitle('Simple menu')
		self.show()


if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = Example()
	sys.exit(app.exec_())