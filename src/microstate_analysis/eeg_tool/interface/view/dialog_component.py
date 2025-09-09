from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QLineEdit, QLabel, QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout, QDialog
from PyQt5.QtCore import Qt
from eeg_tool.interface.model.subject import SubjectInfo, Subject


class OpenFileLabelInput(QtWidgets.QWidget):

    def __init__(self, parent=None, label=None, label_name=None, label_size=None,
                 input=None, input_size=None, button=None, button_name=None):
        super(OpenFileLabelInput, self).__init__(parent)

        self.init_widget(label, label_name, label_size, input, input_size, button, button_name)
        self.layout()
        self.button.clicked.connect(self.click_button)

    def init_widget(self, label=None, label_name=None, label_size=None, input=None, input_size=None,
                    button=None, button_name=None):
        self.label = label
        # self.label.resize(*label_size)
        self.label.setText(label_name)
        self.input = input
        # self.input.resize(*input_size)
        self.button = button
        self.button.setText(button_name)

    def layout(self):
        layout1 = QGridLayout()
        # layout1.setSpacing(100)
        # layout1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout1.addWidget(self.label, 0, 0)
        layout1.addWidget(self.input, 0, 1)
        layout1.addWidget(self.button, 0, 2)
        self.setLayout(layout1)


    def click_button(self):
        self.file_name, self.file_type = QFileDialog.getOpenFileName(self, "Open file", "./", "*(*.*)")
        self.input.setText(self.file_name)

        # subject_info = SubjectInfo(subject_data_path=self.input.setText(file_name), experiment_conf=self.input.setText(file_name),
        #                            condition_name=self.input.setText(file_name), montage_path=self.input.setText(file_name),
        #                            subject_name=self.input.setText(), n_run=self.input.setText())
        # subject = Subject(subject_info)


class LabelInput(QtWidgets.QWidget):
    def __init__(self, parent=None, label=None, label_name=None, input=None):
        super(LabelInput, self).__init__(parent)

        self.init_widget(label, label_name, input)
        self.layout()

    def init_widget(self, label=None, label_name=None, input=None):
        self.label = label
        self.label.setText(label_name)
        self.input = input

    def layout(self):
        layout2 = QGridLayout()
        layout2.addWidget(self.label, 0, 0)
        layout2.addWidget(self.input, 0, 1)
        self.setLayout(layout2)

class LabelComboBox(QtWidgets.QWidget):
    def __init__(self, parent=None, label=None, label_name=None, combobox=None, combobox_items=None):
        super(LabelComboBox, self).__init__(parent)

        self.init_widget(label, label_name, combobox, combobox_items)
        self.layout()

    def init_widget(self, label=None, label_name=None, combobox=None, combobox_items=None):
        self.label = label
        self.label.setText(label_name)
        self.combobox = combobox
        self.combobox.addItems(combobox_items)
        self.combobox.setCurrentIndex(-1)

    def layout(self):
        layout3 = QGridLayout()
        layout3.addWidget(self.label, 0, 0)
        layout3.addWidget(self.combobox, 0, 1)
        self.setLayout(layout3)

class Checkbox(QtWidgets.QWidget):
    def __init__(self, parent=None, checkbox1=None, checkbox_name1=None, checkbox2=None, checkbox_name2=None,
                 checkbox3=None, checkbox_name3=None):
        super(Checkbox, self).__init__(parent)

        self.init_widget(checkbox1, checkbox_name1, checkbox2, checkbox_name2, checkbox3, checkbox_name3)
        self.layout()

    def init_widget(self, checkbox1=None, checkbox_name1=None, checkbox2=None, checkbox_name2=None,
                 checkbox3=None, checkbox_name3=None):
        self.checkbox1 = checkbox1
        self.checkbox1.setText(checkbox_name1)
        self.checkbox2 = checkbox2
        self.checkbox2.setText(checkbox_name2)
        self.checkbox3 = checkbox3
        self.checkbox3.setText(checkbox_name3)

    def layout(self):
        layout4 = QGridLayout()
        layout4.addWidget(self.checkbox1, 0, 0)
        layout4.addWidget(self.checkbox2, 0, 1)
        layout4.addWidget(self.checkbox3, 0, 2)
        self.setLayout(layout4)

class LabelCheckbox(QtWidgets.QWidget):
    def __init__(self, parent=None, checkbox=None, checkbox_name=None, label=None, label_name=None):
        super(LabelCheckbox, self).__init__(parent)

        self.init_widget(checkbox, checkbox_name, label, label_name)
        self.layout()

    def init_widget(self, checkbox=None, checkbox_name=None, label=None, label_name=None):
        self.checkbox = checkbox
        self.checkbox.setText(checkbox_name)
        self.label = label
        self.label.setText(label_name)

    def layout(self):
        layout5 = QGridLayout()
        layout5.addWidget(self.label, 0, 0)
        layout5.addWidget(self.checkbox, 0, 1)
        self.setLayout(layout5)


