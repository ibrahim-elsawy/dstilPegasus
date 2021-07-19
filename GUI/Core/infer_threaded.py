
from PySide2 import QtCore

from Core.inference import InferenceClass


class ThreadClass(QtCore.QThread):
    outputSignal = QtCore.Signal(str)
    errorSignal = QtCore.Signal(str)
    loggingSignal = QtCore.Signal(str)

    def __init__(self, inputs, inference: InferenceClass, parent=None, index=0):
        super(ThreadClass, self).__init__(parent)
        self.index = index
        self.function, self.inputs = inference.infer, inputs
        self.is_running = True

    def run(self):
        print('Starting thread...', self.index)
        try:
            output, elapsed_time, Rouge_Scores = self.function(**self.inputs)
            self.outputSignal.emit(output)
            self.loggingSignal.emit(f' inference time = {elapsed_time} sec, Rouge Score = {Rouge_Scores}')
        except Exception as e:
            self.errorSignal.emit("there was a Problem")
            self.outputSignal.emit('')
            print(e)

    def stop(self):
        self.is_running = False
        print('Stopping thread...', self.index)
        self.terminate()
