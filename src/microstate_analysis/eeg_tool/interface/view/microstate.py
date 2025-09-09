from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class Microstate(QtWidgets.QDialog):
    microstate_signal = pyqtSignal(str, str, int, int, str, str)
    def __init__(self, parent = None):
        super(Microstate,self).__init__(parent)
        self.setWindowTitle('Microstates')
        self.resize(500, 200)
        layout = QGridLayout()

        self.checkbox1 = QCheckBox('Only GFP peaks')
        self.checkbox2 = QCheckBox('Polarity')

        self.label1 = QLabel('Map range:')
        self.label2 = QLabel('Crietia for optimum:')
        self.label3 = QLabel('Method:')
        self.label4 = QLabel('-')

        self.lineEdit1 = QLineEdit('2')
        self.lineEdit2 = QLineEdit('10')

        self.btnPress1 = QPushButton('Reset')
        self.btnPress2 = QPushButton('OK')

        self.combo1 = QComboBox(self)
        self.combo1.addItem('CV')
        self.combo1.addItem('GEV')
        self.combo1.addItem('KL')

        self.combo2 = QComboBox(self)
        self.combo2.addItem('AACH')
        self.combo2.addItem('K-means modified')
        # combo2.currentIndexChanged.connect(self.selection_change)

        layout.addWidget(self.checkbox1, 0, 0)
        layout.addWidget(self.checkbox2, 1, 0)
        layout.addWidget(self.label1, 2, 0)
        layout.addWidget(self.label2, 3, 0)
        layout.addWidget(self.label3, 4, 0)
        layout.addWidget(self.label4, 2, 2)
        layout.addWidget(self.combo1, 3, 1)
        layout.addWidget(self.combo2, 4, 1)
        layout.addWidget(self.lineEdit1, 2, 1)
        layout.addWidget(self.lineEdit2, 2, 3)
        layout.addWidget(self.btnPress1, 5, 4)
        layout.addWidget(self.btnPress2, 5, 5)
        self.setLayout(layout)

        self.btnPress2.clicked.connect(self.btn_ok_clicked)

    def btn_ok_clicked(self):
        check_box1 = str(self.checkbox1.isChecked())
        check_box2 = str(self.checkbox2.isChecked())
        text1 = int(self.lineEdit1.text())
        text2 = int(self.lineEdit2.text())
        combo_box1 = str(self.combo1.currentText())
        combo_box2 = str(self.combo2.currentText())
        self.microstate_signal.emit(check_box1, check_box2, text1, text2, combo_box1, combo_box2)
        self.close()



