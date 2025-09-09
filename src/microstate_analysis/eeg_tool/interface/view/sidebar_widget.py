import sys
import os

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (QTreeView, QHBoxLayout, QVBoxLayout, QApplication, QListView, QMenu, QAction, QFileDialog,
                             QWidget, QTreeWidget, QTreeWidgetItem, QGridLayout, QPushButton, QCheckBox)

from eeg_tool.interface.model.subject import SubjectInfo, Subject
import numpy as np
import mne
import matplotlib
matplotlib.use('QT5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure



class SideBarWidget(QWidget):
    sidebar_signal = pyqtSignal(object)
    sidebar_task_data_signal = pyqtSignal(object)
    sidebar_bad_channel_signal = pyqtSignal(object)
    sidebar_ica_data_signal = pyqtSignal(object)
    sidebar_bad_ica_components_signal = pyqtSignal(object)
    sidebar_artifacts_removal_eeg_signal = pyqtSignal(object)
    sidebar_artifacts_epochs_eeg_signal = pyqtSignal(object)
    sidebar_mark_epochs_signal = pyqtSignal(object)
    def __init__(self, parent=None):
        super(SideBarWidget, self).__init__(parent)
        self.tree_view = QTreeWidget()
        self.tree_view.setColumnCount(1)
        self.tree_view.setHeaderLabels(['Subject'])
        self.root = QTreeWidgetItem(self.tree_view)
        self.tree_view.expandAll()
        self.init_ui()
        # subject_info = SubjectInfo()
        # subject = Subject(subject_info)
        self.tree_view.clicked.connect(self.onClicked)
        # if subject.eeg.ica_data_array is not None:
        #     self.tree_view.clicked.connect(self.open_sec_window)

    def init_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(self.tree_view)
        self.setLayout(layout)

    def add_root(self, file_name):
        self.root.setText(0, file_name)

    def add_child(self, trial_info_array):
        for trial_info in trial_info_array:
            child = QTreeWidgetItem(self.root)
            child.setText(0, trial_info)
            # child.setCheckState(0, Qt.Unchecked)

    def onClicked(self):
        subject_info = SubjectInfo()
        subject = Subject(subject_info)
        # self.task_name = subject.eeg.trial_info
        # self.info = self.tree_view.selectedItems()
        # for ii in self.info:
        #     self.key = ii.text(0)
        #     self.value = self.task_name.get(self.key)[0]
        #     print(self.value)
        #     self.sidebar_signal.emit(self.value)

        self.task_name = subject.eeg.new_trial_info
        self.info = self.tree_view.selectedItems()
        self.name_list = []
        for key in self.task_name:
            self.name_list.append(key)

        if subject.eeg.bad_array is None:
            for ii in self.info:
                self.key = ii.text(0)
                self.value = self.task_name.get(self.key)[0]
                print(self.value)
                self.sidebar_signal.emit(self.value)
        elif subject.eeg.bad_array is not None and subject.eeg.ica_data_array is None:
            self.bad_dict = {}
            self.dic_task_data = {}
            # self.name_list = []
            self.bad_array = subject.eeg.bad_array
            # for key in self.task_name:
            #     self.name_list.append(key)
            for i in range(len(self.bad_array)):
                self.bad_dict.setdefault(self.name_list[i], []).append(self.bad_array[i])
            for i in range(len(subject.eeg.tasks_cleaned)):
                self.dic_task_data.setdefault(self.name_list[i], []).append(subject.eeg.tasks_cleaned[i])

            for ii in self.info:
                self.key = ii.text(0)
                self.bad_value = self.bad_dict.get(self.key)[0]
                self.bad_channel = '; '.join(self.bad_value)
                print(self.bad_channel)
                self.task_value = self.dic_task_data.get(self.key)[0]
                self.sidebar_task_data_signal.emit(self.task_value)
                self.sidebar_bad_channel_signal.emit(self.bad_channel)
        elif subject.eeg.ica_data_array is not None and subject.eeg.mark_epochs is None:
            self.ica_dict = {}
            self.pca_list_dict = {}
            self.ica_list_dict = {}
            self.source_list_dict = {}
            self.mixing_list_dict = {}

            for i in range(len(subject.eeg.ica_data_array)):
                self.ica_dict.setdefault(self.name_list[i], []).append(subject.eeg.ica_data_array[i])
                self.pca_list_dict.setdefault(self.name_list[i], []).append(subject.eeg.pca_list[i])
                self.ica_list_dict.setdefault(self.name_list[i], []).append(subject.eeg.ica_list[i])
                self.source_list_dict.setdefault(self.name_list[i], []).append(subject.eeg.source_list[i])
                self.mixing_list_dict.setdefault(self.name_list[i], []).append(subject.eeg.mixing_list[i])

            for ii in self.info:
                self.key = ii.text(0)
                self.ica_data_value = self.ica_dict.get(self.key)[0]
                self.pca_list_value = self.pca_list_dict.get(self.key)[0]
                self.ica_list_value = self.ica_list_dict.get(self.key)[0]
                self.source_list_value = self.source_list_dict.get(self.key)[0]
                self.mixing_list_value = self.mixing_list_dict.get(self.key)[0]

                self.sidebar_ica_data_signal.emit(self.ica_data_value)

                self.new_window = TopomapWindow(self.pca_list_value, self.ica_list_value, self.source_list_value,
                                                self.mixing_list_value, subject.eeg.rename_channel_dict)
                self.new_window.show()
                self.new_window.checkbox_signal.connect(self.bad_ica_components)
                self.new_window.new_eeg_signal.connect(self.artifacts_removal_eeg)

        elif subject.eeg.mark_epochs is not None:
            self.artifacts_removed_dict = {}
            self.mark_epochs_dict = {}
            self.bad_epochs_dict = {}
            for i in range(len(subject.eeg.artifacts_removal_eeg_list)):
                self.artifacts_removed_dict.setdefault(self.name_list[i], []).append(subject.eeg.artifacts_removal_eeg_list[i])
                self.mark_epochs_dict.setdefault(self.name_list[i], []).append(subject.eeg.mark_epochs[i])
                self.bad_epochs_dict.setdefault(self.name_list[i], []).append(subject.eeg.bad_epochs_list[i])

            for ii in self.info:
                self.key = ii.text(0)
                self.artifacts_removed_value = self.artifacts_removed_dict.get(self.key)[0]
                self.mark_epochs_value = self.mark_epochs_dict.get(self.key)[0]
                self.bad_epochs_value = self.bad_epochs_dict.get(self.key)[0]
                # self.sidebar_artifacts_epochs_eeg_signal.emit(self.artifacts_removed_value)
                self.sidebar_artifacts_epochs_eeg_signal.emit(self.bad_epochs_value)
                self.sidebar_mark_epochs_signal.emit(self.mark_epochs_value)




    def bad_ica_components(self, bad_ica_components):
        subject = Subject()
        subject.eeg.bad_ica_components = bad_ica_components
        self.sidebar_bad_ica_components_signal.emit(subject.eeg.bad_ica_components)

    def artifacts_removal_eeg(self, artifacts_removal_eeg):
        subject = Subject()
        subject.eeg.artifacts_removal_eeg = artifacts_removal_eeg
        self.sidebar_artifacts_removal_eeg_signal.emit(subject.eeg.artifacts_removal_eeg)




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
        # subject = Subject()
        # subject.eeg.bad_ica_components = self.bad_components

    def clicked_trans_button(self):
        subject = Subject()
        self.clean_data = mne.io.RawArray(self.ori_trans.T, subject.eeg.raw_data.info)
        subject.eeg.artifacts_removal_eeg_list.append(self.clean_data)
        self.new_eeg_signal.emit(self.clean_data)
        # subject.eeg.artifacts_removal_eeg = self.clean_data


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