from PyQt5.QtCore import QDate, QTime, QDateTime, Qt
# PyQt5 has currentDate(), currentTime() and currentDateTime() methods for determining current date and time.
# QT提供的3种获取时间日期的方法
now = QDate.currentDate()     # returns the current date
print(now.toString(Qt.ISODate))                   # 2020-01-08
print(now.toString(Qt.DefaultLocaleLongDate))     # 2020年1月8日

datetime = QDateTime.currentDateTime() # returns the current date and time.
print(datetime.toString())             # 周三 1月 8 15:32:39 2020
time = QTime.currentTime()
print(time.toString(Qt.DefaultLocaleLongDate))  # 15:31:52