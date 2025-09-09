import sys
import os
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QMainWindow, QApplication, QDesktopWidget, QMenu, QAction, QFileDialog
import matplotlib

from eeg_tool.interface.model.subject import *

matplotlib.use('QT5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mne.preprocessing.ica import ICA
from eeg_tool.interface.view.sidebar_widget import SideBarWidget
from eeg_tool.interface.view.plot_raw import MainWindowWidget
from eeg_tool.model.eeg import Eeg
from eeg_tool.interface.view.filter_data import FilterData
from eeg_tool.interface.view.microstate import Microstate
from eeg_tool.interface.view.preprocessing.reference_channel import ReferenceChannel
from eeg_tool.interface.view.manual_segmentation import ManualSegmentation
from eeg_tool.interface.view.auto_segmentation import AutoSegmentation
from eeg_tool.interface.view.subject_description import SubjectDescription
from eeg_tool.interface.view.project_description import ProjectDescription
from eeg_tool.interface.view.preprocessing.bad_global_channels import BadGlobalChannels
from eeg_tool.interface.view.preprocessing.bad_epochs import BadEpochs
from eeg_tool.interface.view.preprocessing.artifact_removal import ArtifactRemoval, TopomapWindow
from eeg_tool.interface.view.clustering.individual_eeg_microstates.global_field_power import GlobalFieldPower
from eeg_tool.interface.view.clustering.individual_eeg_microstates.back_fit import BackFit
from eeg_tool.interface.view.clustering.individual_eeg_microstates.plot import Plot
from eeg_tool.interface.view.clustering.individual_eeg_microstates.smooths_microstates_labels import SmoothsMicrostatesLabels
from eeg_tool.interface.view.preprocessing.filtering import Filtering
from eeg_tool.interface.view.clustering.individual_eeg_microstates.microstates_classes import MicrostatesClasses
from eeg_tool.interface.view.clustering.group_eeg_microstates.group_eeg_microstates import GroupEEGMicrostates
from eeg_tool.interface.view.clustering.group_eeg_microstates.full_permutation import FullPermutation
from eeg_tool.interface.view.eeg_microstates import EEGMicrostates


class MainWindow(QMainWindow):
    signal_draw_eeg_parameter = pyqtSignal(object)
    signal_add_root_parameter = pyqtSignal(object)
    signal_add_child_parameter = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_connection()
        # self.eeg = eeg

    def init_connection(self):
        self.open_project_menu.triggered.connect(self.open_file_dialog)
        self.project_data_menu.triggered.connect(self.open_file_dialog)
        self.save_current_data_sets_menu.triggered.connect(self.open_file_dialog)
        # self.filter_signal_menu.triggered.connect(self.open_filter_dialog)
        self.subject_description_menu.triggered.connect(self.subject_description)
        # self.open_file_menu.triggered.connect(self.open_file)
        self.manual_segmentation_menu.triggered.connect(self.manual_segmentation)
        self.auto_segmentation_menu.triggered.connect(self.auto_segmentation)
        self.project_description_menu.triggered.connect(self.project_description)
        self.reference_channel_menu.triggered.connect(self.reference_channel)
        self.filtering_menu.triggered.connect(self.filtering)
        self.bad_global_channels_menu.triggered.connect(self.bad_global_channels)
        self.artifact_removal_menu.triggered.connect(self.artifact_removal)
        self.bad_epochs_menu.triggered.connect(self.bad_epochs)
        self.global_field_power_menu.triggered.connect(self.global_field_power)
        self.microstates_classes_menu.triggered.connect(self.microstates_classes)
        self.back_fit_menu.triggered.connect(self.back_fit)
        self.smooths_microstates_labels_menu.triggered.connect(self.smooths_microstates_labels)
        self.plot_menu.triggered.connect(self.plot)
        self.modified_kmeans_menu.triggered.connect(self.group_eeg_microstates)
        self.full_permutation_menu.triggered.connect(self.full_permutation)
        self.eeg_microstates_menu.triggered.connect(self.eeg_microstates)
        # self.microstate_menu.triggered.connect(self.open_microstate)

    def project_description(self):
        self.description = ProjectDescription()
        self.description.show()

    # def open_file(self):
    #     self.open_file = OpenFile()
    #     self.open_file.signal_draw_eeg_parameter.connect(self.main_window_widget.draw_eeg)
    #     self.open_file.show()

    def subject_description(self):
        self.subject_description = SubjectDescription(self)
        self.subject_description.signal_draw_eeg_parameter.connect(self.main_window_widget.draw_eeg)
        self.subject_description.signal_draw_eeg_parameter.connect(self.side_bar)
        self.subject_description.show()

    def manual_segmentation(self):
        self.manual_segmentation = ManualSegmentation()
        self.project_sidebar_widget.root.takeChildren()
        self.manual_segmentation.signal_draw_eeg_parameter.connect(self.main_window_widget.draw_eeg)
        self.manual_segmentation.signal_draw_eeg_parameter.connect(self.side_bar)
        self.manual_segmentation.show()

    def auto_segmentation(self):
        self.auto_segmentation = AutoSegmentation()
        self.project_sidebar_widget.root.takeChildren()
        self.auto_segmentation.signal_draw_eeg_parameter.connect(self.main_window_widget.draw_eeg)
        self.auto_segmentation.signal_draw_eeg_parameter.connect(self.side_bar)
        self.auto_segmentation.show()


    def reference_channel(self):
        self.reference = ReferenceChannel()
        self.reference.show()

    def filtering(self):
        self.filtering = Filtering(self)
        # self.filtering.filter_signal.connect(self.filter_dialog)
        self.filtering.filter_signal.connect(self.main_window_widget.draw_eeg)
        self.filtering.show()

    # def filter_dialog(self):
        # subject_info = SubjectInfo()
        # subject = Subject(subject_info)
        # subject.eeg.init_preprocessing(raw=subject.eeg.raw_data, tasks=subject.eeg.tasks_data,
        #                                trial_info=subject.eeg.trial_info)
        # subject.eeg.preprocessing.concatenate_tasks()
        # subject.eeg.preprocessing.filter(a, b)
        # subject.eeg.preprocessing.remove_line_noise(c)
        # self.filter_eeg = subject.eeg.preprocessing.tasks_cleaned
        # self.main_window_widget.draw_eeg(self.filter_eeg)

    def side_bar(self):
        subject_info = SubjectInfo()
        subject = Subject(subject_info)
        self.file_name = subject_info.subject_name
        self.task_name = subject.eeg.new_trial_info
        self.project_sidebar_widget.add_root(self.file_name)
        self.project_sidebar_widget.add_child(self.task_name)
        self.project_sidebar_widget.sidebar_signal.connect(self.main_window_widget.mouse_click)
        self.project_sidebar_widget.sidebar_task_data_signal.connect(self.main_window_widget.draw_eeg)
        self.project_sidebar_widget.sidebar_bad_channel_signal.connect(self.main_window_widget.pick_bad_channels)
        self.project_sidebar_widget.sidebar_ica_data_signal.connect(lambda: self.main_window_widget.draw_eeg(
            eeg=self.project_sidebar_widget.ica_data_value, scalings='auto'))
        self.project_sidebar_widget.sidebar_bad_ica_components_signal.connect(self.main_window_widget.pick_bad_channels)
        self.project_sidebar_widget.sidebar_artifacts_removal_eeg_signal.connect(lambda: self.main_window_widget.draw_eeg(
            eeg=subject.eeg.artifacts_removal_eeg, scalings=None))
        self.project_sidebar_widget.sidebar_artifacts_epochs_eeg_signal.connect(self.main_window_widget.draw_eeg)
        self.project_sidebar_widget.sidebar_mark_epochs_signal.connect(self.main_window_widget.draw_vert_line)




    def bad_global_channels(self):
        self.bad_channels = BadGlobalChannels()
        self.bad_channels.select_channels_dialog.manual_bad_channel_signal.\
            connect(self.main_window_widget.pick_bad_channels)
        # self.bad_channels.select_algorithm_dialog.auto_dialog.automatic_bad_channel_signal.\
        #     connect(self.main_window_widget.pick_bad_channels)
        self.bad_channels.show()

    def bad_epochs(self):
        self.bad_epochs = BadEpochs()
        self.bad_epochs.show()
        # self.bad_epochs.epochs_signal.connect(self.main_window_widget.draw_vert_line)

    def artifact_removal(self):
        self.artifact_removal = ArtifactRemoval(self.main_window_widget)
        self.artifact_removal.show()
        self.artifact_removal.ica_signal.connect(lambda: self.main_window_widget.draw_eeg(
            eeg=self.artifact_removal.plot_sources, scalings='auto'))



    def global_field_power(self):
        self.global_field_power = GlobalFieldPower()
        self.global_field_power.show()

    def microstates_classes(self):
        self.microstates_classes = MicrostatesClasses()
        self.microstates_classes.show()

    def back_fit(self):
        self.back_fit = BackFit()
        self.back_fit.show()

    def smooths_microstates_labels(self):
        self.smooths_microstates_labels = SmoothsMicrostatesLabels()
        self.smooths_microstates_labels.show()

    def plot(self):
        self.plot = Plot()
        self.plot.show()

    def group_eeg_microstates(self):
        self.group_eeg_microstates = GroupEEGMicrostates()
        self.group_eeg_microstates.show()

    def full_permutation(self):
        self.full_permutation = FullPermutation()
        self.full_permutation.show()

    def eeg_microstates(self):
        self.eeg_microstates = EEGMicrostates()
        self.eeg_microstates.show()

    def open_filter_dialog(self):
        dialog = FilterData(self)
        dialog.filter_signal.connect(self.deal_inner_slot)
        dialog.show()


    def deal_inner_slot(self, a, b):
        self.current_eeg =self.eeg.raw_eeg.raw_data.filter(a, b)
        # self.current_eeg.raw_eeg.raw_data.plot()

    def open_microstate(self):
        ms = Microstate(self)
        ms.microstate_signal.connect(self.draw_eeg_microstate)
        ms.show()

    def draw_eeg_microstate(self, gfp, polarity, min_maps, max_maps, criteria, method):
        print(gfp, polarity, min_maps, max_maps, criteria, method)


    def init_ui(self):
        self.init_window_size(0.8, 0.8)
        self.center()
        self.setWindowTitle('DesignLab')
        self.init_menubar()
        self.init_layout()
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def init_window_size(self, width, height):
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.resize(self.screenRect.width() * width, self.screenRect.height() * height)

    def init_menubar(self):
        self.menubar = self.menuBar()
        self.init_file_menu()
        self.init_data_analysis_menu()
        self.init_statistical_analysis_menu()
        self.init_research_analysis_menu()
        self.init_help_menu()

    def init_file_menu(self):
        self.file_menubar = self.menubar.addMenu('File')
        self.open_project_menu = QAction('Open project', self)
        self.create_a_new_subject_menu = QMenu('Create a new subject', self)
        self.subject_description_menu = QAction('Subject description', self)

        # self.subject_segmentation_menu = QMenu('Subject segmentation', self)
        # self.open_file_menu = QAction('Open file', self)
        self.manual_segmentation_menu = QAction('Manual segmentation', self)
        self.auto_segmentation_menu = QAction('Auto segmentation', self)

        self.subject_instrument_menu = QAction('Subject instrument', self)
        self.create_a_new_project_menu = QMenu('Create a new project', self)
        self.project_data_menu = QAction('Project data', self)
        self.project_description_menu = QAction('Project description', self)
        self.project_instrument_menu = QMenu('Project instrument', self)
        self.eeg_menu = QAction('EEG', self)
        self.eog_menu = QAction('EOG', self)
        self.ecg_menu = QAction('ECG', self)
        self.emg_menu = QAction('EMG', self)
        self.hrv_menu = QAction('HRV', self)
        self.gsr_menu = QAction('GSR', self)
        self.save_current_data_sets_menu = QAction('Save current data sets', self)
        self.clear_menu = QAction('Clear', self)
        self.quit_menu = QAction('Quit', self)

        self.file_menubar.addAction(self.open_project_menu)
        self.file_menubar.addMenu(self.create_a_new_subject_menu)
        self.create_a_new_subject_menu.addAction(self.subject_description_menu)

        # self.create_a_new_subject_menu.addMenu(self.subject_segmentation_menu)
        # self.subject_segmentation_menu.addAction(self.open_file_menu)
        self.create_a_new_subject_menu.addAction(self.manual_segmentation_menu)
        self.create_a_new_subject_menu.addAction(self.auto_segmentation_menu)

        self.create_a_new_subject_menu.addAction(self.subject_instrument_menu)
        self.file_menubar.addMenu(self.create_a_new_project_menu)
        self.create_a_new_project_menu.addAction(self.project_data_menu)
        self.create_a_new_project_menu.addAction(self.project_description_menu)
        self.create_a_new_project_menu.addMenu(self.project_instrument_menu)
        self.project_instrument_menu.addAction(self.eeg_menu)
        self.project_instrument_menu.addAction(self.eog_menu)
        self.project_instrument_menu.addAction(self.ecg_menu)
        self.project_instrument_menu.addAction(self.emg_menu)
        self.project_instrument_menu.addAction(self.hrv_menu)
        self.project_instrument_menu.addAction(self.gsr_menu)
        self.file_menubar.addAction(self.save_current_data_sets_menu)
        self.file_menubar.addAction(self.clear_menu)
        self.file_menubar.addAction(self.quit_menu)



    def init_data_analysis_menu(self):
        self.data_analysis_menubar = self.menubar.addMenu('Data analysis')
        self.preprocessing_data_menu = QMenu('Preprocessing data', self)
        self.reference_channel_menu = QAction('Reference channel', self)
        self.filtering_menu = QAction('Filtering', self)
        self.bad_global_channels_menu = QAction('Bad global channels', self)
        self.bad_epochs_menu = QAction('Bad epochs', self)
        self.artifact_removal_menu = QAction('Artifact removal',self)
        self.interpolate_and_remove_menu = QAction('Interpolate and remove', self)
        self.classification_menu = QAction('Classification', self)
        self.regression_menu = QAction('Regression', self)
        self.clustering_menu = QMenu('Clustering', self)
        self.individual_eeg_microstates_menu = QMenu('Individual EEG microstates', self)
        self.global_field_power_menu = QAction('Global field power', self)
        self.microstates_classes_menu = QAction('Microstates classes', self)
        self.back_fit_menu = QAction('Back-fit', self)
        self.smooths_microstates_labels_menu = QAction('Smooths microstates labels', self)
        self.plot_menu = QAction('Plot', self)
        self.group_eeg_microstates_menu = QMenu('Group EEG microstates', self)
        self.modified_kmeans_menu = QAction('Modified K-means', self)
        self.full_permutation_menu = QAction('Full permutation', self)

        self.data_analysis_menubar.addMenu(self.preprocessing_data_menu)
        self.preprocessing_data_menu.addAction(self.reference_channel_menu)
        self.preprocessing_data_menu.addAction(self.filtering_menu)
        self.preprocessing_data_menu.addAction(self.bad_global_channels_menu)
        self.preprocessing_data_menu.addAction(self.bad_epochs_menu)
        self.preprocessing_data_menu.addAction(self.artifact_removal_menu)
        self.preprocessing_data_menu.addAction(self.interpolate_and_remove_menu)
        self.data_analysis_menubar.addAction(self.classification_menu)
        self.data_analysis_menubar.addAction(self.regression_menu)
        self.data_analysis_menubar.addMenu(self.clustering_menu)
        self.clustering_menu.addMenu(self.individual_eeg_microstates_menu)
        self.clustering_menu.addMenu(self.group_eeg_microstates_menu)
        self.individual_eeg_microstates_menu.addAction(self.global_field_power_menu)
        self.individual_eeg_microstates_menu.addAction(self.microstates_classes_menu)
        self.individual_eeg_microstates_menu.addAction(self.back_fit_menu)
        self.individual_eeg_microstates_menu.addAction(self.smooths_microstates_labels_menu)
        self.individual_eeg_microstates_menu.addAction(self.plot_menu)
        self.group_eeg_microstates_menu.addAction(self.modified_kmeans_menu)
        self.group_eeg_microstates_menu.addAction(self.full_permutation_menu)





    def init_statistical_analysis_menu(self):
        self.statistical_analysis_menubar = self.menubar.addMenu('Statistical analysis')
        self.preprocessing_menu = QAction('Preprocessing', self)
        self.classification_menu = QAction('Classification', self)
        self.regression_menu = QAction('Regression', self)
        self.clustering_menu = QMenu ('Clustering', self)
        self.eeg_microstates_menu = QAction('EEG microstates', self)

        self.statistical_analysis_menubar.addAction(self.preprocessing_menu)
        self.statistical_analysis_menubar.addAction(self.classification_menu)
        self.statistical_analysis_menubar.addAction(self.regression_menu)
        self.statistical_analysis_menubar.addMenu(self.clustering_menu)
        self.clustering_menu.addAction(self.eeg_microstates_menu)


    def init_research_analysis_menu(self):
        pass


    def init_help_menu(self):
        pass



    def init_layout(self):
        widget = QWidget()
        self.project_sidebar_widget = SideBarWidget(self)
        self.main_window_widget = MainWindowWidget(self)

        layout = QHBoxLayout()
        layout.addWidget(self.project_sidebar_widget)
        layout.addWidget(self.main_window_widget)

        layout.setStretch(0, 2)
        layout.setStretch(1, 8)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

    # def dlg(self):
    #     click_ok
    #     subject_info.data_path = 1

    def open_file_dialog(self):
        # os.getcwd()
        path = r'C:\Users\yaoya\OneDrive\Desktop\eeg\april_02(3)'
        dlg = QFileDialog()
        fname = dlg.getOpenFileName(self, 'Open file', path, 'All Files (*)')[0]
        ext = os.path.splitext(fname)[1][1:].lower()
        eeg_ext = ['vhdr', 'bdf', 'edf']
        sub_ext = ['.txt', 'xlsx']
        loc_ext = ['.locs']
        fmontage = r'C:\Users\yaoya\OneDrive\Documents\eeg_tool\Cap63.locs'
        if ext in eeg_ext:
            # self.eeg.import_eeg(fname, fmontage)
            # self.signal_draw_eeg_parameter.connect(self.main_window_widget.draw_eeg)
            # self.signal_draw_eeg_parameter.emit(self.eeg)
            # fig, params = plot_raw(self.eeg.raw_eeg.raw_data, show=False)
            # f = FigureCanvas(fig)
            # self.setCentralWidget(f)
            # self.main_window_widget.setFocus()
            # # print(fname)
            file_name = fname.split("/")[-1]
            self.signal_add_root_parameter.connect(self.project_sidebar_widget.add_root)
            self.signal_add_root_parameter.emit(file_name)

        elif ext in sub_ext:
            self.eeg.import_trial_info(fname)
            # print([*self.eeg.raw_eeg.trial_info])
            trial_info_array = [*self.eeg.raw_eeg.trial_info]
            self.signal_add_child_parameter.connect(self.project_sidebar_widget.add_child)
            self.signal_add_child_parameter.emit(trial_info_array)

        elif ext in loc_ext:
            self.eeg.import_montage(fname)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
