import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from eeg_tool.interface.view.dialog_component import *

class ProjectDescription(QDialog):

    def __init__(self, parent=None):
        super(ProjectDescription, self).__init__(parent)
        self.setWindowTitle('Project description')
        self.setFixedSize(1600, 600)

        self.component_config = [
            {'label_name': 'Raw data director:', 'label': QLabel(), 'input':QLineEdit(), 'button':QPushButton(),
             'button_name': 'Open file', 'type': 'OpenFileLabelInput'},
            {'label_name': 'Subjects name:', 'label':QLabel(), 'input':QLineEdit(),'button':QPushButton(),
             'button_name': 'Open file', 'type':'OpenFileLabelInput'},
            {'label_name': 'Experiment configure:', 'label':QLabel(),'input':QLineEdit(),'button':QPushButton(),
             'button_name':'Open file', 'type':'OpenFileLabelInput'},
            {'label_name':'The number of run:', 'label':QLabel(),'input':QLineEdit(), 'type':'LabelInput'},
            {'label_name':'Condition name:', 'label': QLabel(), 'input': QLineEdit(), 'type':'LabelInput'}
        ]

        self.component_obj = []
        for item in self.component_config:
            if item['type'] == 'OpenFileLabelInput':
                self.component_obj.append(OpenFileLabelInput(label=item['label'], label_name=item['label_name'],
                                                              input=item['input'], button=item['button'],
                                                              button_name=item['button_name']))
            elif item['type'] == 'LabelInput':
                self.component_obj.append(LabelInput(label=item['label'], label_name=item['label_name'], input=item['input']))


        # self.raw_data_director = OpenFileLabelInput(label=QLabel(), label_name=self.label_name_list[0], label_size=[100,100],
        #                                             input=QLineEdit(), input_size=(20,10), button=QPushButton(), button_name='Open file')
        # self.subjects_name = OpenFileLabelInput(label=QLabel(), label_name='Subjects name:', input=QLineEdit(), button=QPushButton(),
        #                                         button_name='Open file')
        # self.experiment_configure = OpenFileLabelInput(label=QLabel(), label_name='Experiment configure', input=QLineEdit(),
        #                                         button=QPushButton(), button_name='Open file')
        # self.the_number_of_run = LabelInput(label=QLabel(), label_name='The number of run:', input=QLineEdit())
        # self.condition_name = LabelInput(label=QLabel(), label_name='Condition name:', input=QLineEdit())
        self.cancel_button = QPushButton('Cancel', self)
        self.ok_button = QPushButton('OK', self)
        self.set_label_width_by_maximum()
        layout = QGridLayout()

        for i in range(len(self.component_obj)):
            layout.addWidget(self.component_obj[i], i, 0)

        # layout.addWidget(self.raw_data_director, 0, 0)
        # layout.addWidget(self.subjects_name, 1, 0)
        # layout.addWidget(self.experiment_configure, 2, 0)
        # layout.addWidget(self.the_number_of_run,3,0)
        # layout.addWidget(self.condition_name, 4, 0)
        layout.addWidget(self.cancel_button, 5, 1)
        layout.addWidget(self.ok_button, 5, 2)
        self.setLayout(layout)

    def set_label_width_by_maximum(self):
        max_width = 0
        for item in self.component_config:
            width = item['label'].fontMetrics().boundingRect(item['label'].text()).width()
            if max_width < width:
                max_width = width
        for item in self.component_config:
            item['label'].setFixedWidth(max_width)

    # def relabel_name(self):
    #     max_length = 0
    #     for item in self.component_config:
    #         if len(item['label_name']) > max_length:
    #             max_length = len(item['label_name'])
    #     for i in range(len(self.component_config)):
    #         gap = max_length - len(self.component_config[i]['label_name'])
    #         temp = ''.join(['#' for j in range(gap)])
    #         self.component_config[i]['label_name'] = self.component_config[i]['label_name'] + temp
            # for j in range(gap):
            #     item['label_name'] = item['label_name'] + " "


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ProjectDescription()
    win.show()
    sys.exit(app.exec_())