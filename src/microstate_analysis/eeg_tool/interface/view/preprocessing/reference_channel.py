import sys
from PyQt5.QtWidgets import QApplication, QWidget, QCheckBox, QLabel, QComboBox, QPushButton, QGridLayout
from PyQt5.QtCore import Qt, pyqtSignal


class ReferenceChannel(QWidget):
    reference_channel = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.average_reference = QCheckBox('Average reference', self)
        # self.average_reference.move(20, 20)
        # self.average_reference.toggle()
        self.designated_reference = QCheckBox('Designated reference', self)
        # self.designated_reference.move(40,20)
        # self.designated_reference.toggle()

        self.cb = QComboBox()

        self.cb.addItems(['FP1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3',
                          'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1',
                          'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4',
                          'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'FP2', 'AF7',
                          'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1',
                          'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz',
                          'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6',
                          'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8', 'AF8'])
        self.cb.setCurrentIndex(-1)

        self.cancel_button = QPushButton('Cancel')
        self.ok_button = QPushButton('OK')

        layout = QGridLayout()
        layout.addWidget(self.average_reference, 0, 0)
        layout.addWidget(self.designated_reference, 1, 0)
        layout.addWidget(self.cb, 1, 1)
        layout.addWidget(self.cancel_button, 2, 2)
        layout.addWidget(self.ok_button, 2, 3)
        self.setLayout(layout)

        # self.btnPress1.clicked.connect(self.btnPress1_clicked)
        # self.btnPress2.clicked.connect(self.btnPress2_clicked)

        # self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('Reference channel')
        self.show()

if __name__=="__main__":
    import sys
    app=QApplication(sys.argv)
    login_show=ReferenceChannel()
    login_show.show()
    sys.exit(app.exec_())
