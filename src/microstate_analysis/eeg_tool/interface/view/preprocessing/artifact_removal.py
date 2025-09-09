import mne
from PyQt5.QtWidgets import QApplication, QWidget, QCheckBox, QLabel, QComboBox, QPushButton, QGridLayout, QDialog, \
    QLineEdit, QRadioButton, QListWidget, QListWidgetItem, QVBoxLayout, QStackedWidget, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal
from mne.utils import _PCA
from sklearn.decomposition import FastICA, PCA
import matplotlib
matplotlib.use('QT5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from eeg_tool.interface.model.subject import Subject
import numpy as np
from scipy.io import savemat
import matlab
import matlab.engine



class ArtifactRemoval(QWidget):
    eng = matlab.engine.start_matlab()
    ica_signal = pyqtSignal(object)
    mixing_signal = pyqtSignal(object)
    def __init__(self, main_window_widget):
        super().__init__()
        self.main_window_plot_widget = main_window_widget
        self.initUI()

    def initUI(self):
        self.n_components_label = QLabel('n_components:', self)
        self.random_state_label = QLabel('random state:', self)
        self.method_label = QLabel('method:', self)
        self.max_iter_label = QLabel('max_iter:', self)
        self.n_components_input = QLineEdit()
        self.random_state_input = QLineEdit()
        self.method_combo = QComboBox(self)
        self.method_combo.addItems(['Fastica', 'Infomax', 'Picard'])
        self.method_combo.setCurrentIndex(-1)
        self.max_iter_input = QLineEdit()
        self.cancel_button = QPushButton('Cancel')
        self.ok_button = QPushButton('OK')
        self.ok_button.clicked.connect(self.clicked_ok_button)
        # self.ok_button.clicked.connect(self.open_sec_window)

        layout = QGridLayout()
        layout.addWidget(self.n_components_label, 1, 0)
        layout.addWidget(self.n_components_input, 1, 1)
        layout.addWidget(self.random_state_label, 4, 0)
        layout.addWidget(self.random_state_input, 4, 1)
        layout.addWidget(self.method_label, 5, 0)
        layout.addWidget(self.method_combo, 5, 1)
        layout.addWidget(self.max_iter_label, 6, 0)
        layout.addWidget(self.max_iter_input, 6, 1)
        layout.addWidget(self.cancel_button, 12, 3)
        layout.addWidget(self.ok_button, 12, 4)
        self.setLayout(layout)

        self.setWindowTitle('Artifact removal')
        self.show()

    def clicked_ok_button(self):
        n_components = int(self.n_components_input.text())
        max_iter = int(self.max_iter_input.text())

        subject = Subject()
        self.filt_raw = subject.eeg.tasks_cleaned
        self.ica_data = []

        pca_list = []
        ica_list = []
        source_list = []
        mixing_list = []

        for i in range(len(self.filt_raw)):
            self.pca = PCA(n_components=63, whiten=True)
            self.pca_data = self.pca.fit_transform(self.filt_raw[i].get_data().T)
            self.ica = FastICA(n_components=n_components, max_iter=max_iter, tol=0.025, random_state=None, whiten=False)
            self.sources = self.ica.fit_transform(self.pca_data)
            self.plot_sources = mne.io.RawArray(self.sources.T, subject.eeg.raw_data.info)
            self.mixing = self.ica.mixing_

            number = range(0, 63)
            number_list = list(number)
            num_str = [str(j) for j in number_list]
            new_list = []
            for k in num_str:
                new_list.append('ICA' + k)
            rename = zip(subject.eeg.raw_data.ch_names, new_list)
            self.rename_channel = dict(rename)
            self.plot_sources.rename_channels(self.rename_channel)
            self.ica_data.append(self.plot_sources)

            pca_list.append(self.pca)
            ica_list.append(self.ica)
            source_list.append(self.sources)
            mixing_list.append(self.mixing)
            if i >= 2:
                break

        subject.eeg.ica_data_array = self.ica_data
        subject.eeg.pca_list = pca_list
        subject.eeg.ica_list = ica_list
        subject.eeg.source_list = source_list
        subject.eeg.mixing_list = mixing_list
        subject.eeg.rename_channel_dict = self.rename_channel
        # self.ica_signal.emit(self.plot_sources)
        self.close()

    # def clicked_ok_button(self):
    #     n_components = int(self.n_components_input.text())
    #     max_iter = int(self.max_iter_input.text())
    #
    #     subject = Subject()
    #     self.filt_raw = subject.eeg.preprocessing.tasks_cleaned
    #
    #     self.pca = PCA(n_components=63, whiten=True)
    #     self.pca_data = self.pca.fit_transform(self.filt_raw.get_data().T)
    #     self.ica = FastICA(n_components=n_components, max_iter=max_iter, tol=0.025, random_state=None, whiten=False)
    #     self.sources = self.ica.fit_transform(self.pca_data)
    #     self.plot_sources = mne.io.RawArray(self.sources.T, subject.eeg.raw_data.info)
    #
    #     number = range(0, 63)
    #     number_list = list(number)
    #     num_str = [str(i) for i in number_list]
    #     new_list = []
    #     for i in num_str:
    #         new_list.append('ICA' + i)
    #     rename = zip(subject.eeg.raw_data.ch_names, new_list)
    #     self.rename_channel = dict(rename)
    #     self.plot_sources.rename_channels(self.rename_channel)
    #     self.ica_signal.emit(self.plot_sources)
    #     # self.mixing_signal.emit(self.mixing[0])
    #     self.close()

    def open_sec_window(self):
        # self.mixing = self.ica.mixing_
        self.new_window = TopomapWindow(self.pca, self.ica, self.sources, self.mixing, self.rename_channel)
        # self.mat_data = matlab.double(self.filt_raw.get_data().tolist())
        # ArtifactRemoval.eng.edit('MARA', nargout=0)
        # self.auto_artifacts = ArtifactRemoval.eng.MARA(self.mat_data)
        # print(self.auto_artifacts)
        # print(type(self.auto_artifacts))
        # print(self.auto_artifacts.shape)
        self.new_window.checkbox_signal.connect(self.main_window_plot_widget.pick_bad_channels)
        self.new_window.new_eeg_signal.connect(lambda: self.main_window_plot_widget.draw_eeg(
            eeg=self.new_window.clean_data, scalings=None))
        self.new_window.show()


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


