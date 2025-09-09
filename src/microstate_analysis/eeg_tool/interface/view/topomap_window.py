from eeg_tool.interface.model.subject import SubjectInfo, Subject
import numpy as np
import mne
import matplotlib
matplotlib.use('QT5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (QTreeView, QHBoxLayout, QVBoxLayout, QApplication, QListView, QMenu, QAction, QFileDialog,
                             QWidget, QTreeWidget, QTreeWidgetItem, QGridLayout, QPushButton, QCheckBox)

class TopomapWindow(QWidget):
    checkbox_signal = pyqtSignal(object)
    new_eeg_signal = pyqtSignal(object)
    def __init__(self, pca, ica, sources, mixing, rename_channel):
        super().__init__()
        self.pca = pca
        self.ica = ica
        self.sources = sources
        self.mixing = mixing
        self.rename_channel = rename_channel
        self.channel_value = list(self.rename_channel.values())
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('DesignLab')
        self.init_layout()
        self.resize(2500, 1900)

    def init_layout(self):
        self.layout = QGridLayout()
        self.layout.setSpacing(15)
        self.trans_button = QPushButton('Transform')
        self.cancel_button = QPushButton('Cancel')
        self.layout1 = QHBoxLayout()
        self.layout1.addStretch()
        self.layout1.addWidget(self.trans_button)
        self.layout1.addWidget(self.cancel_button)
        self.layoutall = QVBoxLayout()
        self.layoutall.addLayout(self.layout)
        self.layoutall.addLayout(self.layout1)
        self.setLayout(self.layoutall)
        subject = Subject()
        self.draw(data=self.mixing, pos=subject.eeg.raw_data.info)
        for i in range(len(self.map)):
            self.map[i].checkBox.stateChanged.connect(self.on_checkbox)
        self.trans_button.clicked.connect(self.clicked_trans_button)
        self.cancel_button.clicked.connect(self.close)

    def on_checkbox(self):
        bad_ica = []
        sel_list = []
        for i in range(len(self.map)):
            if self.map[i].checkBox.isChecked():
                bad_ica.append(self.map[i].checkBox.text())
                sel_list.append(i)
        self.bad_components = '; '.join(bad_ica)
        self.new_mixing = np.delete(self.mixing, sel_list, axis=1)
        self.new_fit_trans = np.delete(self.sources, sel_list, axis=1)
        self.ica_inverse_trans = np.dot(self.new_fit_trans, self.new_mixing.T)
        self.ori_trans = np.dot(self.ica_inverse_trans, np.sqrt(self.pca.explained_variance_[:, np.newaxis]) *
                            self.pca.components_) + self.pca.mean_

        self.checkbox_signal.emit(self.bad_components)

    def clicked_trans_button(self):
        subject = Subject()
        self.clean_data = mne.io.RawArray(self.ori_trans.T, subject.eeg.raw_data.info)
        self.new_eeg_signal.emit(self.clean_data)

    def draw(self, data, pos):
        self.map = {}
        for i in range(len(data)):
            self.map[i] = PhotoWidget(self)
            self.layout.addWidget(self.map[i], i/8, i % 8)
            self.map[i].checkBox.setText(self.channel_value[i])
            ax = self.map[i].fig.add_subplot(111)
            mne.viz.plot_topomap(data[i], pos, show=False, axes=ax, image_interp='spline36', contours=6)


class PhotoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.checkBox = QCheckBox(self)
        self.fig = Figure()
        self.topomap_figure = FigureCanvas(self.fig)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.checkBox)
        self.layout.addWidget(self.topomap_figure)
        self.setLayout(self.layout)