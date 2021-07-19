import sys
from PySide2 import QtGui, QtCore
from PySide2.QtWidgets import (QMainWindow,
                               QPushButton, QWidget, QLabel,
                               QHBoxLayout, QVBoxLayout, QCheckBox,
                               QProgressBar, QListWidget, QListWidgetItem,
                               QTextEdit, )
from PySide2.QtGui import QFont, QIcon
from PySide2.QtCore import Qt

from Core.infer_threaded import ThreadClass
from Core.inference import InferenceClass, modelspaths

import pathlib as path

class EmittingStream(QtCore.QObject):
    textWritten = QtCore.Signal(str)

    def __init__(self, textWrittenFunction):
        QtCore.QObject.__init__(self)
        self.textWritten.connect(textWrittenFunction)

    def write(self, text):
        self.textWritten.emit(str(text))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()


        # Log Signal
        sys.stdout = EmittingStream(textWrittenFunction=self.normalOutputWritten)

        # inference class
        self.inference_class = InferenceClass()

        # components
        self.Title = QLabel('Hello\n Please Choose a Model and Enter the text Passage to be summarized.')
        self.Title.setMargin(20)
        self.Title.setAlignment(Qt.AlignHCenter)
        self.Title.setFont(QFont('Ariel', 16, QFont.DemiBold))

        self.listWidget = QListWidget()

        items = list(modelspaths.keys())
        for item_text in items:
            item = QListWidgetItem(item_text)
            self.listWidget.addItem(item)
        item = self.listWidget.item(0)
        self.listWidget.setCurrentItem(item)

        self.quantizedCheckbox = QCheckBox('Quantized')

        self.inputField = QTextEdit()
        t_input = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
        self.inputField.setPlaceholderText(t_input)
        self.inputField.setFont(QFont('Ariel', 16, QFont.DemiBold))

        self.outputLabel = QLabel("Summary:")
        self.outputLabel.hide()
        self.outputLabel.setFont(QFont('Ariel', 12, QFont.ExtraBold))
        self.outputField = QTextEdit()
        self.outputField.setDocumentTitle("Summary")
        self.outputField.hide()
        self.outputField.setFont(QFont('Ariel', 14, QFont.ExtraBold))

        self.button = QPushButton('Summarize')
        self.button.clicked.connect(lambda: self.onClicked())

        self.progress = QProgressBar()
        self.progress.hide()

        self.logLabel = QLabel()
        # self.logLabel.hide()
        self.logLabel.setText("                                                ")
        self.logLabel.setFont(QFont('Ariel', 12, QFont.ExtraBold))

        self.textEdit = QTextEdit()
        self.textEdit.hide()

        self.window = QWidget()
        self.window.setWindowIcon(QtGui.QIcon('screens\\icon.png'))
        self.window.resize(800, 800)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.Title)

        self.modelLayout = QHBoxLayout()
        self.modelLayout.addWidget(self.listWidget)
        self.modelLayout.addWidget(self.quantizedCheckbox)

        self.LoggingLayout = QVBoxLayout()
        self.LoggingLayout_inner = QHBoxLayout()
        self.showLogsButton = QPushButton('Show Logs')
        self.showLogsButton.clicked.connect(lambda: self.showLogs())
        self.LoggingLayout_inner.addWidget(self.logLabel)
        self.LoggingLayout_inner.addWidget(self.showLogsButton)
        self.LoggingLayout.addLayout(self.LoggingLayout_inner)
        self.LoggingLayout.addWidget(self.textEdit)

        self.layout.addLayout(self.modelLayout)

        self.layout.addWidget(self.inputField)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.progress)
        self.layout.addWidget(self.outputLabel)
        self.layout.addWidget(self.outputField)
        self.layout.addLayout(self.LoggingLayout)
        self.window.setLayout(self.layout)

        self.textEditVisibility = False

    def __del__(self):
        # Restore sys.stdout
        sys.stdout = sys.__stdout__

    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.textEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textEdit.setTextCursor(cursor)
        self.textEdit.ensureCursorVisible()

    def showLogs(self):
        self.textEdit.setVisible(not self.textEditVisibility)
        self.textEditVisibility = not self.textEditVisibility
        if self.textEditVisibility:
            self.showLogsButton.setText('Hide Logs')
        else:
            self.showLogsButton.setText('Show Logs')

    def show(self):
        self.window.show()

    def onClicked(self):
        self.progress.show()
        self.progress.setMinimum(0)
        self.progress.setMaximum(0)
        self.button.setDisabled(True)
        text = self.inputField.toPlainText()
        text = self.inputField.placeholderText() if text == '' else text
        item = self.listWidget.currentItem()
        checked = self.quantizedCheckbox.isChecked()
        self.mythread = ThreadClass({'text': text, 'models': item, 'quantized': checked}, self.inference_class)
        self.mythread.outputSignal.connect(self.showOutput)
        self.mythread.errorSignal.connect(self.printLogs)
        self.mythread.loggingSignal.connect(self.printLogs)
        self.mythread.start()

    def showOutput(self, output):
        self.outputField.show()
        self.outputLabel.show()
        self.outputField.setText(output if output != '' else 'please enter proper input')
        self.progress.hide()
        self.button.setDisabled(False)
        self.mythread.stop()

    def printLogs(self, logMsg):
        self.logLabel.show()
        self.logLabel.setText(logMsg)
