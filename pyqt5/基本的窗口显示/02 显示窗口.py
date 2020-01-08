import sys
from PyQt5.QtWidgets import QApplication, QWidget
# 改代码用于显示一个窗口
if __name__ == '__main__':
	app = QApplication(sys.argv)

	w = QWidget()
	w.resize(250, 150)
	w.move(300, 300)
	w.setWindowTitle('Simple')
	w.show()

	sys.exit(app.exec_())