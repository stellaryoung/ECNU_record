import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QApplication

# 按键盘escape键，应用会退出
# Events in PyQt5 are processed often by reimplementing event handlers.
class Example(QWidget):

	def __init__(self):
		super().__init__()

		self.initUI()

	def initUI(self):
		self.setGeometry(300, 300, 250, 150)
		self.setWindowTitle('Event handler')
		self.show()

	def keyPressEvent(self, e):              # reimplement the keyPressEvent() event handler.
		if e.key() == Qt.Key_Escape:
			self.close()


if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = Example()
	sys.exit(app.exec_())

if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = Example()
	sys.exit(app.exec_())