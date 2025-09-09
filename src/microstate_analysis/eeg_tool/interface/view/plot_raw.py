from functools import partial
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QStackedWidget
import matplotlib
from mne.viz.raw import _plot_update_raw_proj, _plot_raw_traces

matplotlib.use('QT5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mne.viz.utils import _plot_raw_onscroll, _plot_raw_onkey, _mouse_click, _plot_raw_time
from eeg_tool.interface.view.plot_raw_component import plot_raw
import numpy as np
import mne

class MainWindowWidget(QWidget):
    def __init__(self, parent):
        super(MainWindowWidget, self).__init__(parent)
        layout = QVBoxLayout(self)
        self.stackWidget = QStackedWidget()
        layout.addWidget(self.stackWidget)
        self.setLayout(layout)

    def draw_eeg(self, eeg, scalings=None):
        fig, self.params = plot_raw(eeg, show=False, scalings=scalings)
        self.draw_eeg_figure = FigureCanvas(fig)
        self.draw_eeg_figure.mpl_connect('scroll_event', partial(_plot_raw_onscroll, params=self.params))
        self.draw_eeg_figure.mpl_connect('key_press_event', partial(_plot_raw_onkey, params=self.params))
        self.draw_eeg_figure.mpl_connect('button_press_event', partial(_mouse_click, params=self.params))
        self.stackWidget.addWidget(self.draw_eeg_figure)
        self.stackWidget.setCurrentIndex(self.stackWidget.count() - 1)

    def pick_bad_channels(self, bads):
        temp = bads.split(';')
        temp_data = []
        for i in temp:
            temp_data.append(i.replace(" ",""))
        self.params['info']['bads'] = temp_data
        _plot_update_raw_proj(self.params, None)

    def mouse_click(self, xdata):
        _plot_raw_time(float(xdata) * 0.002, self.params)
        self.params['update_fun']()
        self.params['plot_fun']()

    def draw_vert_line(self, xdata):
        ax = self.params['ax']
        vertline_color = (0., 0.75, 0.)
        for i in range(len(xdata)):
            axdata = float(xdata[i]) * 0.002
            self.params['ax_vertline'] = ax.axvline(0, color=vertline_color, zorder=4)
            self.params['ax_vertline'].set_xdata(axdata)
            self.params['ax_hscroll_vertline'].set_xdata(axdata)
            self.params['vertline_t'].set_text('%0.2f  ' % axdata)




