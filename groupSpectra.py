# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:37:04 2020

@author: xbrjos
"""
from PyQt5 import QtWidgets, QtCore
import os
import numpy as np
import pickle

from functions import read_format_and_save_spectra
from viewitems import SpectrumView, PCAClusterView, ResultSpectra
defaultPath = r'C:\Users\xbrjos\Desktop\Unsynced Files\Weathered FTIR Spectra\processed by MH Febr2020'


class MainView(QtWidgets.QWidget):
    """
    The Main View for running the FTIR Clustering.
    """
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

        self.pcaClusteringPlot = PCAClusterView(self)
        self.dirPath = None
        self._establish_connections()

    def _establish_connections(self):
        self.spectraContainer.spectraHaveChanged.connect(self.pcaClusteringPlot.update_all)
        self.spectraContainer.spectraHaveChanged.connect(self.spectraPlots.update_display)
        self.spectraContainer.spectraSelectionHasChanged.connect(self.pcaClusteringPlot.update_all)
        self.spectraPlots.spectraOptionsChanged.connect(self.spectraContainer.update_spectra_options)
        self.pcaClusteringPlot.spectrumIndexSelected.connect(self.spectraPlots.jump_to_spec_index)

    def selectSpectraFolder(self) -> None:
        """
        This is the starting point for loading a set of spectra.
        All spectra have to be as csv in the selected directory.
        :return:
        """
        def has_file(files: list, fileextension: str) -> bool:
            hasFile: bool = False
            for curFile in files:
                if curFile.endswith(fileextension):
                    hasFile = True
                    break
            return hasFile
        
        self.dirPath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Spectra Directory', defaultPath)
        if self.dirPath:
            projectName = os.path.basename(self.dirPath)
            self.setWindowTitle(f'Clustering of Spectra of {projectName}')
            spectra = None
            allFiles = os.listdir(self.dirPath)
            for file in allFiles:
                if file.endswith('.npy'):
                    spectra = np.load(os.path.join(self.dirPath, file))
                    break
            if spectra is None:
                spectra = read_format_and_save_spectra(self.dirPath, allFiles)

            self._reset_specContainer_with_spectra(spectra)

            if has_file(allFiles, 'pkl'):
                with open(os.path.join(self.dirPath, 'selectedSpectra.pkl'), "rb") as fp:
                    selectedSpectraIndice: list = pickle.load(fp)
                self.spectraContainer.update_selected_spectra_from_list(selectedSpectraIndice)

            self._initialize_child_windows()

        else:
            self.dirPath = None

    def _reset_specContainer_with_spectra(self, spectra: np.array) -> None:
        """
        When a new set of spectra is loaded, te spectraContainer needs to be resetted with these.
        :param spectra:
        :return:
        """
        self.spectraContainer.clear_all_spectra()
        for index in range(spectra.shape[1]-1):
            spec: np.array = np.transpose(np.vstack((spectra[:, 0], spectra[:, index+1])))
            self.spectraContainer.add_spectrum(spec, index)

    def _initialize_child_windows(self):
        self.showMaximized()
        self.spectraPlots.currentPageIndex = 0
        self.spectraPlots.go_to_page(0)
        self.pcaClusteringPlot.update_all()
        self.pcaClusteringPlot.show()

    def closeEvent(self, event) -> None:
        if self.dirPath is not None:
            self.spectraContainer.save_selected_spectraIndices(self.dirPath)
        self.pcaClusteringPlot.close()
        event.accept()


class SpectraContainer(QtCore.QObject):
    """
    The SpectraContainer keeps all SpectraObjects (i.e., GUI version of the spectrum data) and acts as an interface
    for all other objects to interact with the actual spectral data, but also their graphical representation.
    """
    spectraHaveChanged = QtCore.pyqtSignal()
    spectraSelectionHasChanged = QtCore.pyqtSignal()

    def __init__(self):
        super(SpectraContainer, self).__init__()
        self.spectraObjects: list = []

    def clear_all_spectra(self) -> None:
        """
        Clears all references to previously stored spectra.
        :return:
        """
        self.spectraObjects = []

    def add_spectrum(self, spectrum: np.array, specIndex: int) -> None:
        """
        Adds a spectrum in an np.array (1st Col: Wavenumbers, 2nd Col, Intensities) and registers it with a
        unique spectrumIndex.
        :param spectrum:
        :param specIndex:
        :return:
        """
        newSpecView: SpectrumView = SpectrumView(self, spectrum, specIndex)
        self.spectraObjects.append(newSpecView)

    def get_number_of_spectra(self) -> int:
        """
        Returns the number of stored spectra.
        :return:
        """
        return len(self.spectraObjects)

    def get_number_of_selected_spectra(self) -> int:
        """
        Returns the number of spectra that are currently selected.
        :return:
        """
        numSelected: int = 0
        for specObj in self.spectraObjects:
            if specObj.isSelected:
                numSelected += 1
        return numSelected

    def get_widget_of_spectrum_of_index(self, index: int) -> SpectrumView:
        """
        Returns reference to the SpecView belonging to the indicated specIndex
        :param index:
        :return:
        """
        specView: SpectrumView = self.spectraObjects[index]
        specView.update_specGraph()
        return specView

    def get_selected_spectra(self) -> np.array:
        """
        Returns an Array (1st Col: Wavenumber, 2nd-nth Col: Intensities) of all spectra that are currently selected.
        :return:
        """
        selectedSpectra: list = []
        for index, specObj in enumerate(self.spectraObjects):
            if index == 0:
                selectedSpectra.append(specObj.spectrum[:, 0])

            if specObj.isSelected:
                selectedSpectra.append(specObj.spectrum[:, 1])
        spectraArray: np.array = np.transpose(np.array(selectedSpectra))
        return spectraArray

    def update_spec_selection(self):
        self.spectraSelectionHasChanged.emit()

    @QtCore.pyqtSlot(bool, bool)
    def update_spectra_options(self, subtractBaseline: bool, removeCO2: bool):
        """
        Enforces recalculation of all spectra according to given parameters.
        :param subtractBaseline:
        :param removeCO2:
        :return:
        """
        for specObj in self.spectraObjects:
            specObj.update_spectra_options(subtractBaseline, removeCO2)
        self.spectraHaveChanged.emit()

    def update_selected_spectra_from_list(self, selectedSpectraIndice: list) -> None:
        """
        Takes a list of indices of spectra that were selected in a previous session and updates
        selection of the currently created specObjects to match that selection.
        :param selectedSpectraIndice:
        :return:
        """
        for specObj in self.spectraObjects:
            if specObj.specIndex in selectedSpectraIndice:
                specObj.isSelected = True
            else:
                specObj.isSelected = False
            specObj.update_opacity()

    def save_selected_spectraIndices(self, path: str) -> None:
        """
        Saves list of indices of currently selected spectra to file in order to recover selection in a later session.
        :param path:
        :return:
        """
        selectedSpectraIndice: list = []
        for specObj in self.spectraObjects:
            if specObj.isSelected:
                selectedSpectraIndice.append(specObj.specIndex)

        with open(os.path.join(path, 'selectedSpectra.pkl'), "wb") as fp:
            pickle.dump(selectedSpectraIndice, fp, protocol=-1)

    def highlight_spectrum_of_index(self, specIndex: int) -> None:
        """
        The indicated spectrum is highlighted for easier visibility
        :param specIndex:
        :return:
        """
        for specObj in self.spectraObjects:
            if specObj.specIndex == specIndex:
                specObj.set_highlight()
            else:
                specObj.remove_highlight()


class SpectraPlotViewer(QtWidgets.QGroupBox):
    """
    Displays an array of SpectraObjects (i.e., QGraphicViews) in a grid Layout and handles navigation between the pages.
    """
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
        self.baseLineCheckBox.stateChanged.connect(self.announce_changed_options)
        self.removeCO2CheckBox.stateChanged.connect(self.announce_changed_options)
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
        """
        Makes the current display show all spectra belonging to the page with the given index.
        :param pageIndex:
        :return:
        """
        self.currentPageIndex = pageIndex
        self._remove_currently_displayed_spectra()
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
        """
        Updates the current display.
        :return:
        """
        self._update_pageLabel()
        self._update_currently_displayed_spectra()

    @QtCore.pyqtSlot(int)
    def jump_to_spec_index(self, specIndex: int) -> None:
        """
        Given a spec Index, the corresponding page is displayed and the spectrum is highlighted.
        :param specIndex:
        :return:
        """
        numSpecsPerPage: int = self.numCols * self.numRows
        pageIndex: int = specIndex // numSpecsPerPage
        self.go_to_page(pageIndex)
        self.spectraContainer.highlight_spectrum_of_index(specIndex)

    def _update_pageLabel(self) -> None:
        """
        A QLabel informs the user on what page he/she currently is. It is updated here.
        :return:
        """
        numSpectra = self.spectraContainer.get_number_of_spectra()
        numSpectraSelected = self.spectraContainer.get_number_of_selected_spectra()
        self.pageLabel.setText(f'(page {self.currentPageIndex + 1} of {self.numPages};'
                               f' {numSpectraSelected} of {numSpectra} spectra selected)')

    def _update_currently_displayed_spectra(self) -> None:
        """
        Forces an update of all currently displayed spectra.
        :return:
        """
        for specObj in self.currentlyDisplayedSpectra:
            specObj.update_specGraph()

    def announce_changed_options(self):
        """
        Is called, when a spectra-Processing option is changed.
        :return:
        """
        self.spectraOptionsChanged.emit(self.baseLineCheckBox.isChecked(), self.removeCO2CheckBox.isChecked())

    def _remove_currently_displayed_spectra(self) -> None:
        """
        The curretly displayed specObjects are removed fom the layout so that new ones can be added.
        :return:
        """
        for spec in self.currentlyDisplayedSpectra:
            spec.setParent(None)
            self.spectraGroupLayout.removeWidget(spec)
        self.currentlyDisplayedSpectra = []

    def _update_buttons(self) -> None:
        """
        The navigation buttons are enabled/disabled according to total page number.
        :return:
        """
        self.prevPgeBtn.setDisabled(self.currentPageIndex == 0)
        self.nextPgeBtn.setDisabled(self.currentPageIndex == self.numPages-1)

    def _to_previous_page(self) -> None:
        """
        Makes the ui go one page pack.
        :return:
        """
        self.currentPageIndex -= 1
        self._update_buttons()
        self.go_to_page(self.currentPageIndex)

    def _to_next_page(self) -> None:
        """
        Makes the ui to advance one page.
        :return:
        """
        self.currentPageIndex += 1
        self._update_buttons()
        self.go_to_page(self.currentPageIndex)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainView()
    ret = app.exec_()
