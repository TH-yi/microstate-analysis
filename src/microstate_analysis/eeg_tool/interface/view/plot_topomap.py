import mne
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QStackedWidget
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QT5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# class TopomapWidget(QWidget):
#     def __init__(self, parent):
#         super(TopomapWidget, self).__init__(parent)
#         self.layout = QVBoxLayout(self)
#         self.stackWidget = QStackedWidget()
#         self.layout.addWidget(self.stackWidget)
#         self.setLayout(self.layout)
#
#     def draw_topomap(self, row, col, data, pos):
#         self.fig = Figure()
#         for i in range(row):
#             for j in range(col):
#                 ax = self.fig.add_subplot(row, col, i * col + j + 1)
#                 mne.viz.plot_topomap(data[i*col+j], pos, show=False, axes=ax, image_interp='spline36', contours=6)
#                 if i * col + j + 1 == len(data):
#                     break
#             if i * col >= len(data):
#                 break
#         # plt.subplots_adjust(hspace=0.4, wspace=0.4)
#         self.topomap_figure = FigureCanvas(self.fig)
#         self.stackWidget.addWidget(self.topomap_figure)
#         self.stackWidget.setCurrentIndex(self.stackWidget.count() - 1)

class TopomapWindow(QWidget):
    def __init__(self, mixing):
        super().__init__()
        self.mixing = mixing
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('DesignLab')
        self.init_layout()
        self.resize(1600, 1000)

    def init_layout(self):
        self.layout = QVBoxLayout(self)
        self.stackWidget = QStackedWidget()
        self.layout.addWidget(self.stackWidget)
        self.setLayout(self.layout)

    def topomap(self):
        subject = Subject()
        self.draw_topomap(data=self.mixing, pos=subject.eeg.raw_data.info, row=8, col=8)

    def draw_topomap(self, row, col, data, pos):
        self.fig = Figure()
        for i in range(row):
            for j in range(col):
                ax = self.fig.add_subplot(row, col, i * col + j + 1)
                mne.viz.plot_topomap(data[i*col+j], pos, show=False, axes=ax, image_interp='spline36', contours=6)
                if i * col + j + 1 == len(data):
                    break
            if i * col >= len(data):
                break
        self.topomap_figure = FigureCanvas(self.fig)
        self.stackWidget.addWidget(self.topomap_figure)
        self.stackWidget.setCurrentIndex(self.stackWidget.count() - 1)