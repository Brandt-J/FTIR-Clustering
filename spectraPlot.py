"""FTIR Spectra Clustering
Copyright (C) 2020 Josef Brandt,
University of Gothenborg <josef.brandt@gu.se>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program, see COPYING.
If not, see <https://www.gnu.org/licenses/>.
"""
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np

from refDatabase import ReferenceDatabase


class SpectraPlot(QtWidgets.QGroupBox):
    refSelectionChanged = QtCore.pyqtSlot()

    def __init__(self, refDatabase: ReferenceDatabase) -> None:
        super(SpectraPlot, self).__init__()
        layout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        self.setFlat(True)

        self.refDatabase: ReferenceDatabase = refDatabase

        self.specCanvas: FigureCanvas = FigureCanvas(Figure())
        self.specAx = self.specCanvas.figure.add_subplot(111)
        self.refAx = self.specAx.twinx()

        self.navToolBarWidth: int = 50
        self.navToolBar: NavigationToolbar = NavigationToolbar(self.specCanvas, self)
        self.navToolBar.setOrientation(QtCore.Qt.Vertical)
        self.navToolBar.setFixedWidth(self.navToolBarWidth)

        layout.addWidget(self.navToolBar)
        layout.addWidget(self.specCanvas)

    def update_sample_spectra(self, spectraList: list, plotTitle: str, specLabels: list = None) -> None:
        """
        Updates the spectra given in the list.
        :param spectraList: The list containes np.array, where each array has the wavenumbers in
        first col and an arbitrary number of cols for intensities.
        :param plotTitle: Title to show in the plot
        :param specLabels: Nested list of labels for each spectrum to show
        :return:
        """
        self.specAx.clear()
        for outerIndex, spectra in enumerate(spectraList):
            for innerIndex, specInd in enumerate(range(spectra.shape[1] - 1)):
                specLabel: str = (specLabels[outerIndex][innerIndex] if specLabels is not None else '')
                self.specAx.plot(spectra[:, 0], spectra[:, specInd + 1], label=specLabel)
                self.specAx.set_title(plotTitle)
                self.specAx.set_xlabel('Wavenumber (cm-1)')
                self.specAx.set_ylabel('Abundancy (a.u.)')
        if specLabels is not None:
            self.specAx.legend(loc='upper left')
        self.specCanvas.draw()

    def update_reference_spectrum(self, refName: str, refSpec: np.array) -> None:
        """
        Draws a the spectrum of the given reference on the second axis.
        :param refName:
        :param refSpec:
        :return:
        """
        self.refAx.clear()
        if refName != '':
            self.refAx.plot(refSpec[:, 0], refSpec[:, 1], label=refName, color='r')
            self.refAx.legend(loc='upper right')
        self.specCanvas.draw()

    def set_canvas_width(self, widgetWidthPx: int) -> None:
        canvasWidthPx: float = widgetWidthPx - self.navToolBarWidth
        canvasWidthInch: float = canvasWidthPx / self.logicalDpiX()
        fig: Figure = self.specCanvas.figure
        fig.set_figwidth(canvasWidthInch)
