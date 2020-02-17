# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:37:04 2020

@author: xbrjos
"""
from PyQt5 import QtWidgets, QtCore
import os
import numpy as np

from functions import read_format_and_save_spectra
from viewitems import SpectrumView, PCAClusterView, ResultSpectra
defaultPath = r'C:\Users\xbrjos\Desktop\Unsynced Files\Weathered FTIR Spectra\processed by MH Febr2020'


class MainView(QtWidgets.QWidget):
    def __init__(self):
        super(MainView, self).__init__()
        self.setWindowTitle('FTIR Clustering')
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        self.openBtn = QtWidgets.QPushButton('Select Spectra Folder')
        self.openBtn.released.connect(self.selectSpectraFolder)
        self.openBtn.setFixedWidth(150)
        layout.addWidget(self.openBtn)

        self.spectraContainer: SpectraContainer = SpectraContainer()
        self.spectraPlots: SpectraPlotViewer = SpectraPlotViewer(self.spectraContainer)
        layout.addWidget(self.spectraPlots)
        self.show()

        self.pcaClusteringPlot = PCAClusterView(self.spectraContainer)
        self.spectraContainer.spectraHaveChanged.connect(self.pcaClusteringPlot.update_all)
        self.spectraContainer.spectraHaveChanged.connect(self.spectraPlots.update_display)
        self.spectraContainer.spectraSelectionHasChanged.connect(self.pcaClusteringPlot.update_all)
        self.spectraPlots.spectraOptionsChanged.connect(self.spectraContainer.update_spectra_options)

    def selectSpectraFolder(self):
        dirpath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Spectra Directory', defaultPath)
        if dirpath:
            self.spectraContainer.clear_all_spectra()
            spectra = None
            allSpecFiles = os.listdir(dirpath)
            for file in allSpecFiles:
                if file.find('.npy') != -1:
                    spectra = np.load(os.path.join(dirpath, file))
                    break
            if spectra is None:
                spectra = read_format_and_save_spectra(dirpath, allSpecFiles)

            for index in range(spectra.shape[1]-1):
                spec: np.array = np.transpose(np.vstack((spectra[:, 0], spectra[:, index+1])))
                self.spectraContainer.add_spectrum(spec, index)

            self.showMaximized()
            self.spectraPlots.currentPageIndex = 0
            self.spectraPlots.go_to_page(0)
            self.pcaClusteringPlot.update_all()
            self.pcaClusteringPlot.show()

    def closeEvent(self, event) -> None:
        self.pcaClusteringPlot.close()
        event.accept()


class SpectraContainer(QtCore.QObject):
    spectraHaveChanged = QtCore.pyqtSignal()
    spectraSelectionHasChanged = QtCore.pyqtSignal()

    def __init__(self):
        super(SpectraContainer, self).__init__()
        self.spectraObjects: list = []

    def clear_all_spectra(self) -> None:
        self.spectraObjects = []

    def add_spectrum(self, spectrum: np.array, specIndex: int) -> None:
        self.spectraObjects.append(SpectrumView(self, spectrum, specIndex))

    def get_number_of_spectra(self) -> int:
        return len(self.spectraObjects)

    def get_number_of_selected_spectra(self) -> int:
        numSelected: int = 0
        for specObj in self.spectraObjects:
            if specObj.isSelected:
                numSelected += 1
        return numSelected

    def get_widget_of_spectrum_of_index(self, index: int) -> SpectrumView:
        specView: SpectrumView = self.spectraObjects[index]
        specView.update_specGraph()
        return specView

    def get_selected_spectra(self) -> np.array:
        selectedSpectra: list = []
        for index, specObj in enumerate(self.spectraObjects):
            if index == 0:
                selectedSpectra.append(specObj.spectrum[:, 0])

            if specObj.isSelected:
                selectedSpectra.append(specObj.spectrum[:, 1])
        spectraArray: np.array = np.transpose(np.array(selectedSpectra))
        return spectraArray

    def update_spec_selection(self):
        # self.spectraHaveChanged.emit()
        self.spectraSelectionHasChanged.emit()

    @QtCore.pyqtSlot(bool, bool)
    def update_spectra_options(self, subtractBaseline: bool, removeCO2: bool):
        for specObj in self.spectraObjects:
            specObj.update_spectra_options(subtractBaseline, removeCO2)
        self.spectraHaveChanged.emit()


class SpectraPlotViewer(QtWidgets.QGroupBox):
    spectraOptionsChanged = QtCore.pyqtSignal(bool, bool)

    def __init__(self, spectraContainer: SpectraContainer, plotShape: tuple = (4, 4)):
        super(SpectraPlotViewer, self).__init__()

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        
        controlLayout = QtWidgets.QHBoxLayout()
        controlGroup = QtWidgets.QGroupBox()
        controlGroup.setLayout(controlLayout)
        layout.addWidget(controlGroup)
        
        self.spectraContainer: SpectraContainer = spectraContainer
        self.numRows: int = plotShape[0]
        self.numCols: int = plotShape[1]

        navigationGroup = QtWidgets.QGroupBox('Navigation')
        navigationLayout = QtWidgets.QHBoxLayout()
        navigationGroup.setLayout(navigationLayout)

        self.prevPgeBtn: QtWidgets.QPushButton = QtWidgets.QPushButton('Previous Page')
        self.prevPgeBtn.setFixedWidth(100)
        self.prevPgeBtn.released.connect(self._to_previous_page)
        self.nextPgeBtn: QtWidgets.QPushButton = QtWidgets.QPushButton('Next Page')
        self.nextPgeBtn.setFixedWidth(100)
        self.nextPgeBtn.released.connect(self._to_next_page)
        self.pageLabel = QtWidgets.QLabel()
        navigationLayout.addWidget(self.prevPgeBtn)
        navigationLayout.addWidget(self.nextPgeBtn)
        navigationLayout.addWidget(self.pageLabel)

        optionsGroup = QtWidgets.QGroupBox('Spectra Options')
        optionsLayout = QtWidgets.QHBoxLayout()
        optionsGroup.setLayout(optionsLayout)

        self.baseLineCheckBox = QtWidgets.QCheckBox('Subtract Baseline')
        self.removeCO2CheckBox = QtWidgets.QCheckBox('Remove CO2 region')
        self.baseLineCheckBox.stateChanged.connect(self.anounceChangedOptions)
        self.removeCO2CheckBox.stateChanged.connect(self.anounceChangedOptions)
        optionsLayout.addWidget(self.baseLineCheckBox)
        optionsLayout.addWidget(self.removeCO2CheckBox)

        controlLayout.addWidget(navigationGroup)
        controlLayout.addWidget(optionsGroup)

        spectraGroup = QtWidgets.QGroupBox('Spectra View')
        self.spectraGroupLayout = QtWidgets.QGridLayout()
        spectraGroup.setLayout(self.spectraGroupLayout)
        layout.addWidget(spectraGroup)

        self.currentlyDisplayedSpectra: list = []
        self.currentPageIndex: int = 0

    @property
    def numPages(self) -> int:
        return int(np.ceil(self.spectraContainer.get_number_of_spectra() / (self.numCols * self.numRows)))

    def go_to_page(self, pageIndex: int = 0) -> None:
        self._reset_currently_displayed_spectra()
        self.update_display()
        startSpecIndex: int = pageIndex * self.numRows * self.numCols
        specCounter: int = 0
        maxSpecIndex: int = self.spectraContainer.get_number_of_spectra() - 1

        for rowInd in range(self.numRows):
            for colInd in range(self.numCols):
                specIndex = startSpecIndex + specCounter
                if specIndex <= maxSpecIndex:
                    widget: SpectrumView = self.spectraContainer.get_widget_of_spectrum_of_index(specIndex)
                    specCounter += 1
                else:
                    widget: SpectrumView = SpectrumView(self)

                self.currentlyDisplayedSpectra.append(widget)
                self.spectraGroupLayout.addWidget(widget, rowInd, colInd)

    def update_display(self) -> None:
        self._update_pageLabel()
        self._update_currently_displayed_spectra()

    def _update_pageLabel(self) -> None:
        numSpectra = self.spectraContainer.get_number_of_spectra()
        numSpectraSelected = self.spectraContainer.get_number_of_selected_spectra()
        self.pageLabel.setText(f'(page {self.currentPageIndex + 1} of {self.numPages};'
                               f' {numSpectraSelected} of {numSpectra} spectra selected)')

    def _update_currently_displayed_spectra(self) -> None:
        for specObj in self.currentlyDisplayedSpectra:
            specObj.update_specGraph()

    def anounceChangedOptions(self):
        self.spectraOptionsChanged.emit(self.baseLineCheckBox.isChecked(), self.removeCO2CheckBox.isChecked())

    def _reset_currently_displayed_spectra(self) -> None:
        for spec in self.currentlyDisplayedSpectra:
            spec.setParent(None)
            self.spectraGroupLayout.removeWidget(spec)
        self.currentlyDisplayedSpectra = []

    def _update_buttons(self) -> None:
        self.prevPgeBtn.setDisabled(self.currentPageIndex == 0)
        self.nextPgeBtn.setDisabled(self.currentPageIndex == self.numPages-1)

    def _to_previous_page(self) -> None:
        self.currentPageIndex -= 1
        self._update_buttons()
        self.go_to_page(self.currentPageIndex)

    def _to_next_page(self) -> None:
        self.currentPageIndex += 1
        self._update_buttons()
        self.go_to_page(self.currentPageIndex)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainView()
    ret = app.exec_()
