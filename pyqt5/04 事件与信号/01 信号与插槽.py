import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QLCDNumber, QSlider,
							 QVBoxLayout, QApplication)


class Example(QWidget):

	def __init__(self):
		super().__init__()

		self.initUI()

	def initUI(self):
		lcd = QLCDNumber(self)              # 定义显示数字的
		sld = QSlider(Qt.Horizontal, self)  # 定义滑动的按钮组件

		vbox = QVBoxLayout()
		vbox.addWidget(lcd)
		vbox.addWidget(sld)

		self.setLayout(vbox)
		sld.valueChanged.connect(lcd.display)   # 将lcd（sender）与sld(receiver)进行关联

		self.setGeometry(300, 300, 250, 150)
		self.setWindowTitle('Signal and slot')
		self.show()


if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = Example()
	sys.exit(app.exec_())