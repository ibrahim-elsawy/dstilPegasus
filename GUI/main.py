# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import logging
import sys
import os
import faulthandler

from PySide2.QtWidgets import QApplication
from qt_material import apply_stylesheet

from GUI.MainWindow import MainWindow


logging.basicConfig(level=logging.ERROR)
faulthandler.is_enabled()
os.environ["PYTHONFAULTHANDLER"] = '1'
os.environ["PYTHONASYNCIODEBUG"] = '0'
os.environ["QT_SCREEN_SCALE_FACTORS"] = '1'
os.environ["QT_SCALE_FACTOR"] = '1'
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = '0'


def main():
    form = MainWindow()
    form.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationDisplayName("ATS")

    apply_stylesheet(app, 'dark_yellow.xml')

    main()

    sys.exit(app.exec_())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
