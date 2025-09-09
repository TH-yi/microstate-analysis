from PyQt5.QtWidgets import QApplication, QWidget, QCheckBox, QLabel, QComboBox, QPushButton, QGridLayout, QDialog, \
    QRadioButton, QListWidget, QListWidgetItem
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5 import QtWidgets
from eeg_tool.interface.view.dialog_component import *
from eeg_tool.interface.model.subject import *
global bad_channel
bad_channel = []
class BadGlobalChannels(QWidget):

    def __init__(self):
        super().__init__()
        # self.auto_bad_channel = auto_bad_channel
        self.select_channels_dialog = SelectChannelDialog()
        self.select_algorithm_dialog = SelectAlgorithmDialog()
        self.initUI()

    def initUI(self):
        self.manual = QRadioButton('Manual', self)
        self.automatic = QRadioButton('Automatic', self)
        self.cancel_button = QPushButton('Cancel')
        self.ok_button = QPushButton('OK')

        layout = QGridLayout()
        layout.addWidget(self.manual, 0, 0)
        layout.addWidget(self.automatic, 1, 0)
        layout.addWidget(self.cancel_button, 2, 2)
        layout.addWidget(self.ok_button, 2, 3)
        self.setLayout(layout)

        self.setWindowTitle('Bad global channels')
        self.show()
        self.ok_button.clicked.connect(self.select_channels)

    def select_channels(self):
        if self.manual.isChecked():
            self.select_channels_dialog.show()
            self.select_channels_dialog.exec()
            self.close()
        elif self.automatic.isChecked():
            self.select_algorithm_dialog.show()
            self.select_algorithm_dialog.exec()
            self.close()

class SelectAlgorithmDialog(QDialog):
    def __init__(self, parent=None):
        super(SelectAlgorithmDialog, self).__init__(parent)
        # self.auto_bad_channel = auto_bad_channel
        self.bad_array = []
        self.setWindowTitle('Select algorithm')
        self.algorithm_label = QLabel('Select algorithm:', self)
        self.prep_pipeline = QRadioButton('Prep_pipeline', self)
        self.cancel_button = QPushButton('Cancel')
        self.ok_button = QPushButton('OK')
        # self.auto_dialog = AutoDialog()
        self.ok_button.clicked.connect(self.select_algorithm)
        self.ok_button.clicked.connect(self.open_auto_dialog)

        layout = QGridLayout()
        layout.addWidget(self.algorithm_label, 0, 0)
        layout.addWidget(self.prep_pipeline, 0, 1)
        layout.addWidget(self.cancel_button, 2, 2)
        layout.addWidget(self.ok_button, 2, 3)
        self.setLayout(layout)



    def select_algorithm(self):
        if self.prep_pipeline.isChecked():
            subject = Subject()
            self.auto_bad_channel = []
            for i in range(len(subject.eeg.tasks_cleaned)):
                subject.eeg.preprocessing.remove_bad_channel(thread=5, threshold=0.1,
                                                             filtered_data=subject.eeg.tasks_cleaned[i])
                self.auto_bads = subject.eeg.preprocessing.global_bads
                self.bad_array.append(self.auto_bads)
                self.auto_bad_channel += self.auto_bads
                subject.eeg.global_good_index_list.append(subject.eeg.preprocessing.global_good_index)
                if i >= 2:
                    break

            subject.eeg.bad_array = self.bad_array
            print(subject.eeg.bad_array)
            print(type(subject.eeg.bad_array))
            # self.auto_dialog.a.setBadChannelArray(self.auto_bad_channel)
            # self.auto_dialog.setBadChannelArray(self.auto_bad_channel)
            # self.auto_dialog.show()
            # self.auto_dialog.exec()
            self.close()
            # subject = Subject()
            # subject.eeg.preprocessing.remove_bad_channel(thread=5, threshold=0.1, filtered_data=subject.eeg.tasks_cleaned)
            # self.auto_bad_channel = subject.eeg.preprocessing.global_bads
            #
            # self.auto_dialog.setBadChannelArray(self.auto_bad_channel)
            # self.auto_dialog.show()
            # self.auto_dialog.exec()
            # self.close()

    def open_auto_dialog(self):
        self.auto_dialog = AutoDialog(self.bad_array)
        self.auto_dialog.show()


class AutoDialog(QDialog):
    automatic_bad_channel_signal = pyqtSignal(object)
    def __init__(self, bads):
        super(AutoDialog, self).__init__()
        self.bads = bads
        self.select_channels_dialog = SelectChannelDialog()
        self.setWindowTitle('Automatic')
        self.cancel_button = QPushButton('Cancel', self)
        self.ok_button = QPushButton('OK', self)
        self.ok_button.clicked.connect(self.clicked_ok_button)

        layout1 = QHBoxLayout()
        layout1.addStretch(1)
        layout1.addWidget(self.cancel_button)
        layout1.addWidget(self.ok_button)
        self.layout2 = QVBoxLayout()
        self.layout2.addStretch(1)
        self.list_layout(bads=self.bads)
        self.layout2.addLayout(layout1)
        self.setLayout(self.layout2)
        for i in range(len(self.map)):
            self.map[i].button.clicked.connect(self.selected)

    def list_layout(self, bads):
        self.map = {}
        subject = Subject()
        task_name = list(subject.eeg.new_trial_info.keys())
        for i in range(len(bads)):
            self.map[i] = LabelList(self)
            self.layout2.addWidget(self.map[i])
            self.map[i].task_label.setText(task_name[i])
            # self.bad_channel_list = self.map[i].listwidget
            # self.bad_channel_label = self.map[i].bad_label
            self.map[i].button.setText('select')
            for oneValue in bads[i]:
                elem = QListWidgetItem(oneValue, self.map[i].listwidget)
                elem.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                elem.setCheckState(Qt.Checked)

    def selected(self):
        for i in range(len(self.map)):
            count = self.map[i].listwidget.count()
            j = 0
            channel_list = []
            while j < count:
                item = self.map[i].listwidget.item(j)
                if item.checkState() == Qt.Checked:
                    channel_list.append(item.text())
                j += 1
            self.bad_list_label = '; '.join(channel_list)
            self.map[i].bad_label.clear()
            self.map[i].bad_label.setText(self.bad_list_label)


    # def get_selected(self):
    #     count = self.bad_channel_list.count()
    #     i = 0
    #     channel_list = []
    #     while i < count:
    #         item = self.bad_channel_list.item(i)
    #         if item.checkState() == Qt.Checked:
    #             channel_list.append(item.text())
    #         i += 1
    #         return channel_list
    #
    # def show_selected(self):
    #     self.bad_channel_label.clear()
    #     self.channel_list = '; '.join(self.get_selected())
    #     self.bad_channel_label.setText(self.channel_list)

        # subject = Subject()
        # task_name = list(subject.eeg.new_trial_info.keys())
        # self.component_config = []
        # self.bad_channel_list = QListWidget(self)
        # self.select_button = QPushButton(self)
        # self.bad_channel_label = QLabel(self)
        # self.bad_channel_label.setWordWrap(True)
        #
        # for i in range(len(subject.eeg.new_trial_info)):
        #     self.component_config.append(
        #         {'label_name': task_name[i], 'label': QLabel(), 'qlist': self.bad_channel_list,
        #          'button': self.select_button, 'button_name': 'Select',
        #          'label1': self.bad_channel_label, 'type': 'LabelList'}
        #     )
        #     if i >= 2:
        #         break
        # layout = QGridLayout()
        # self.component_obj = []
        # for item in self.component_config:
        #     if item['type'] == 'LabelList':
        #         self.component_obj.append(
        #             LabelList(label=item['label'], label_name=item['label_name'], qlist=item['qlist'],
        #                       button=item['button'], button_name=item['button_name'], label1=item['label1']))
        # for childwidget in self.component_obj:
        #     i = self.component_obj.index(childwidget)
        #     layout.addWidget(childwidget, i, 0)
        # self.setLayout(layout)

        # self.select_button.clicked.connect(self.show_selected)



        # layout1 = QHBoxLayout()
        # layout1.addStretch(1)
        # layout1.addWidget(self.cancel_button)
        # layout1.addWidget(self.ok_button)
        # layout2 = QVBoxLayout()
        # layout2.addStretch(1)
        # # self.a = ListView(self)
        # # layout2.addWidget(self.a)
        # layout2.addLayout(layout)
        # layout2.addLayout(layout1)
        #
        # self.setLayout(layout2)


    def clicked_ok_button(self):
        global bad_channel
        # self.auto_bads = self.bad_channel_label.text()
        # print(type(self.auto_bads))
        # print(self.auto_bads)
        # print(type(bad_channel))
        # print(bad_channel)
        # bad_channel = bad_channel + '; ' + self.auto_bads
        print(bad_channel)
        # self.automatic_bad_channel_signal.emit(bad_channel)
        self.close()

    # def get_selected(self):
    #     count = self.bad_channel_list.count()
    #     i = 0
    #     channel_list = []
    #     while i < count:
    #         item = self.bad_channel_list.item(i)
    #         if item.checkState() == Qt.Checked:
    #             channel_list.append(item.text())
    #         i += 1
    #     return channel_list
    #
    # def show_selected(self):
    #     self.bad_channel_label.clear()
    #     self.channel_list = '; '.join(self.get_selected())
    #     self.bad_channel_label.setText(self.channel_list)


    # def setBadChannelArray(self, arr):
    #     for oneValue in arr:
    #         elem = QListWidgetItem(oneValue, self.bad_channel_list)
    #         elem.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
    #         elem.setCheckState(Qt.Checked)


class LabelList(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.task_label = QLabel()
        self.listwidget = QListWidget()
        self.button = QPushButton()
        self.bad_label = QLabel()

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.task_label)
        self.layout.addWidget(self.listwidget)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.bad_label)
        self.setLayout(self.layout)



class SelectChannelDialog(QDialog):
    manual_bad_channel_signal = pyqtSignal(object)
    def __init__(self, parent=None):
        super(SelectChannelDialog, self).__init__(parent)
        self.setWindowTitle('Manual')
        self.manual_label = QLabel('Manual', self)
        subject = Subject()
        self.ch_names = subject.eeg.raw_data.ch_names
        self.combo = ComboCheckBox(self.ch_names)
        self.cancel_button = QPushButton('Cancel')
        self.ok_button = QPushButton('OK')

        layout = QGridLayout()
        layout.addWidget(self.manual_label, 0, 0)
        layout.addWidget(self.combo, 0, 1)
        layout.addWidget(self.cancel_button, 2, 2)
        layout.addWidget(self.ok_button, 2, 3)
        self.setLayout(layout)

        self.ok_button.clicked.connect(self.clicked_ok_button)


    def clicked_ok_button(self):
        global bad_channel
        bad_channel = self.combo.currentText()
        self.manual_bad_channel_signal.emit(bad_channel)
        self.close()



class ComboCheckBox(QComboBox):
    def __init__(self, items: list):
        super(ComboCheckBox, self).__init__()
        self.items = items
        self.box_list = []
        self.text = QLineEdit()
        # self.state = 0

        q = QListWidget()
        for i in range(len(self.items)):
            self.box_list.append(QCheckBox())
            self.box_list[i].setText(self.items[i])
            item = QListWidgetItem(q)
            q.setItemWidget(item, self.box_list[i])
            self.box_list[i].stateChanged.connect(self.show_selected)

        q.setStyleSheet("font-size: 20px; font-weight: bold; height: 40px; margin-left: 5px")
        self.setStyleSheet("width: 300px; height: 50px; font-size: 21px; font-weight: bold")
        self.text.setReadOnly(True)
        self.setLineEdit(self.text)
        self.setModel(q.model())
        self.setView(q)

    def get_selected(self) -> list:
        ret = []
        for i in range(0, len(self.items)):
            if self.box_list[i].isChecked():
                ret.append(self.box_list[i].text())
        return ret

    def show_selected(self):
        self.text.clear()
        ret = '; '.join(self.get_selected())
        self.text.setText(ret)







if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    login_show = BadGlobalChannels()
    login_show.show()
    sys.exit(app.exec_())
