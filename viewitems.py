"""
FTIR Spectra Clustering
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
from matplotlib.backend_bases import PickEvent
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import os
from itertools import combinations

from clustering import SpectraCluster
import functions as fn
from spectraPlot import SpectraPlot
from refDatabase import ReferenceDatabase


class SpectrumView(QtWidgets.QGraphicsView):
    """
    A spectrum is rendered as pixmap and can be shown in any QWidget.
    The original spectrum can be modified with different methods (baseline, co2-removal)
    """
    def __init__(self, parentContainer, spectrum: np.array = None, specIndex: int = None):
        super(SpectrumView, self).__init__()
        self.parentContainer = parentContainer
        self.origSpectrum: np.array = spectrum
        self.spectrum = None
        if self.origSpectrum is not None:
            self.spectrum = self.origSpectrum.copy()
        self.specIndex: int = specIndex

        self.drag = None

        self.setHorizontalScrollBarPolicy(1)
        self.setVerticalScrollBarPolicy(1)

        self.highlightTimer: QtCore.QTimer = QtCore.QTimer()
        self.highlightTimer.setSingleShot(True)
        self.highlightTimer.setInterval(2000)
        self.highlightTimer.timeout.connect(self.remove_highlight)

        scene = QtWidgets.QGraphicsScene(self)
        scene.setItemIndexMethod(QtWidgets.QGraphicsScene.NoIndex)
        scene.setBackgroundBrush(QtCore.Qt.darkGray)
        self.setScene(scene)
        self.setCacheMode(QtWidgets.QGraphicsView.CacheBackground)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.BoundingRectViewportUpdate)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        self.item = QtWidgets.QGraphicsPixmapItem()
        self.item.setPos(0, 0)

        self.scene().addItem(self.item)
        self.item.setScale(0.15)

        self.isSelected: bool = True
        self._do_auto_deselection()

    def update_spectra_options(self, subtractBaseline: bool, removeCO2: bool):
        """
        Subtracts baseline and/or removes CO2 region.
        :param subtractBaseline:
        :param removeCO2:
        :return:
        """
        if self.origSpectrum is not None:
            self.spectrum = self.origSpectrum.copy()
            if subtractBaseline:
                self.spectrum[:, 1] -= fn.get_baseline(self.spectrum[:, 1], smoothness_param=1e6)

            if removeCO2:
                self.spectrum = fn.remove_co2(self.spectrum)
        else:
            self.spectrum = None

    def update_specGraph(self) -> None:
        """
        Updates the pixmap for display.
        :return:
        """
        newPixmap: QtGui.QPixmap = self._spectrum_to_pixmap(self.spectrum)
        self.item.setPixmap(newPixmap)

    def _do_auto_deselection(self) -> None:
        """
        According to a simple noise-level-approximation, poor spectra are automatically deselected.
        :return:
        """
        if self.spectrum is None:
            self.isSelected = False
        else:
            noiseLevel: float = fn.get_noise_level(self.spectrum[:, 1])
            if noiseLevel > 3:
                self.isSelected = False
        self.update_opacity()

    def _spectrum_to_pixmap(self, spectrum: np.array) -> QtGui.QPixmap:
        """
        Converts a spectrum into a pixmap.
        :param spectrum:
        :return:
        """
        # sp = SubplotParams(left=0., bottom=0., right=1., top=1.)
        fig = Figure(figsize=(8, 4), dpi=300)
        canvas: FigureCanvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        if spectrum is not None:
            ax.plot(spectrum[:, 0], spectrum[:, 1])
            ax.set_title(f'Spectrum number {self.specIndex+1}')
        ax.set_ylabel('Absorption')
        ax.set_xlabel('Wavenumber (cm-1)')
        canvas.draw()
        width, height = canvas.size().width(), canvas.size().height()
        im = QtGui.QImage(canvas.buffer_rgba(), width, height, QtGui.QImage.Format_ARGB32)
        return QtGui.QPixmap(im)

    def set_highlight(self) -> None:
        """
        Initiates a highlighting effect to make the specView shine out of a collection of multiple instances.
        :return:
        """
        self.scene().setBackgroundBrush(QtCore.Qt.green)
        self.highlightTimer.start()

    def remove_highlight(self) -> None:
        """
        Removes the highlighting effect.
        :return:
        """
        self.scene().setBackgroundBrush(QtCore.Qt.darkGray)

    def update_opacity(self) -> None:
        """
        Updates the pixmap's opacity to indicate selection or deselection, respectively.
        :return:
        """
        if self.isSelected:
            self.item.setOpacity(1.)
        else:
            self.item.setOpacity(0.2)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """
        Used for handling dragging the graph and/or selecting the spectrum.
        :param event:
        :return:
        """
        if event.button() == QtCore.Qt.MiddleButton:
            self.drag = event.pos()

        elif event.button() == QtCore.Qt.LeftButton:
            self.isSelected = not self.isSelected
            self.update_opacity()
            self.parentContainer.update_spec_selection()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """
        Used for dragging the graphView.
        :param event:
        :return:
        """
        if self.drag is not None:
            p0 = event.pos()
            move = self.drag - p0
            self.horizontalScrollBar().setValue(move.x() + self.horizontalScrollBar().value())
            self.verticalScrollBar().setValue(move.y() + self.verticalScrollBar().value())

            self.drag = p0

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self.drag = None

    def wheelEvent(self, event):
        """
        Used for zoom.
        :param event:
        :return:
        """
        factor: float = 1.01 ** (event.angleDelta().y() / 8)
        self.item.setScale(self.item.scale() * factor)


class PCAClusterView(QtWidgets.QWidget):
    """
    Graphical View of the PCA clustering, containing the clustered PCA data and the explained variance plot.
    """
    spectrumIndexSelected = QtCore.pyqtSignal(int)

    def __init__(self, mainWinParent):
        super(PCAClusterView, self).__init__()

        self.mainWinParent = mainWinParent
        self.spectraContainer = mainWinParent.spectraContainer
        self.specClusterer: SpectraCluster = SpectraCluster(self.spectraContainer)

        self.setWindowTitle('PCA Clustering')
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        toolbar = QtWidgets.QGroupBox('Clustering Controls')
        toolbarLayout = QtWidgets.QHBoxLayout()
        toolbar.setLayout(toolbarLayout)
        toolbar.setMaximumHeight(100)

        self.pc1Selector = QtWidgets.QSpinBox()
        self.pc2Selector = QtWidgets.QSpinBox()
        self.pc3Selector = QtWidgets.QSpinBox()
        for index, spinbox in enumerate([self.pc1Selector, self.pc2Selector, self.pc3Selector], 1):
            spinbox.setMinimum(1)
            spinbox.setMaximum(self.specClusterer.highestComponent)
            spinbox.setValue(index)
            spinbox.setFixedWidth(50)
            spinbox.valueChanged.connect(self.update_all)

        self.pc3CheckBox = QtWidgets.QCheckBox('Include 3rd dimension with PC Comp')
        self.pc3CheckBox.stateChanged.connect(self.update_all)

        self.numClusterSelector = QtWidgets.QSpinBox()
        self.numClusterSelector.setMinimum(1)
        self.numClusterSelector.setMaximum(7)
        self.numClusterSelector.setValue(2)
        self.numClusterSelector.setFixedWidth(70)
        self.numClusterSelector.valueChanged.connect(self.update_all)

        # self.optimizeBtn = QtWidgets.QPushButton('Optimize')
        # self.optimizeBtn.released.connect(self._optimize_params)

        toolbarLayout.addWidget(QtWidgets.QLabel('PC Components to plot:'))
        toolbarLayout.addWidget(self.pc1Selector)
        toolbarLayout.addWidget(self.pc2Selector)
        toolbarLayout.addWidget(self.pc3CheckBox)
        toolbarLayout.addWidget(self.pc3Selector)
        toolbarLayout.addWidget(QtWidgets.QLabel(' | Num. Clusters in PCA:'))
        toolbarLayout.addWidget(self.numClusterSelector)
        # toolbarLayout.addWidget(QtWidgets.QLabel('|'))
        # toolbarLayout.addWidget(self.optimizeBtn)

        updateSpecsBtn = QtWidgets.QPushButton('Update Spectra View')
        updateSpecsBtn.released.connect(self.update_result_spectra)

        toolbarLayout.addWidget(QtWidgets.QLabel(' |'))
        toolbarLayout.addWidget(updateSpecsBtn)
        toolbarLayout.addStretch()

        layout.addWidget(toolbar)
        self.panelLayout = QtWidgets.QHBoxLayout()

        self.pcaCanvas3d = FigureCanvas(Figure())
        self.pcAx3d = self.pcaCanvas3d.figure.add_subplot(111, projection='3d')

        self.pcaCanvas2d = FigureCanvas(Figure())
        self.pcAx2d = self.pcaCanvas2d.figure.add_subplot(111)
        self.varCanvas = FigureCanvas(Figure())
        self.var_ax = self.varCanvas.figure.add_subplot(111)
        self.pca2DNavigation = NavigationToolbar(self.pcaCanvas2d, self)
        self.pca2DNavigation.setOrientation(QtCore.Qt.Vertical)
        self.pca2DNavigation.setFixedWidth(50)

        self.canvasSplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addLayout(self.panelLayout)

        self.resultSpectraPlot = ResultSpectra(self.mainWinParent, self)
        self._toggle_3d_clustering()

    def reset_for_new_sample(self):
        """
        Called when a new sample is loaded.
        :return:
        """
        self.update_all()
        self.show()
        self.update_result_spectra()

    def update_all(self) -> None:
        """
        Updates the specClusterer and all displays.
        :return:
        """
        spectra: np.array = self.spectraContainer.get_selected_spectra()
        numSpectra: int = spectra.shape[1] - 1
        self._check_for_highest_possible_comps(numSpectra)
        self.specClusterer.spectra = spectra
        self._toggle_3d_clustering()

        clusterComponents = None
        if not self.pc3CheckBox.isChecked():
            clusterComponents = (self.pc1Selector.value()-1, self.pc2Selector.value()-1, None)
        else:
            clusterComponents = (self.pc1Selector.value()-1, self.pc2Selector.value()-1, self.pc3Selector.value()-1)

        self.specClusterer.clusterComponents = clusterComponents
        self.specClusterer.numDesiredClusters = self.numClusterSelector.value()

        self.specClusterer.update()
        self._update_cluster_plot()
        self._update_variance_plot()

    def _toggle_3d_clustering(self) -> None:
        """
        According to the selection of the third princ. comp., the canvasses are exchanged
        and the pc3-spinbox is enabled/disabled.
        :return:
        """
        self.pc3Selector.setEnabled(self.pc3CheckBox.isChecked())
        self.pcaCanvas3d.setParent(None)
        self.pcaCanvas2d.setParent(None)
        self.varCanvas.setParent(None)
        self.pca2DNavigation.setParent(None)
        if self.pc3CheckBox.isChecked():
            self.canvasSplitter.addWidget(self.pcaCanvas3d)
            self.pcaCanvas3d.draw()
        else:
            self.panelLayout.addWidget(self.pca2DNavigation)
            self.canvasSplitter.addWidget(self.pcaCanvas2d)
            self.pcaCanvas2d.draw()

        self.canvasSplitter.addWidget(self.varCanvas)
        self.canvasSplitter.setSizes([1, 1])

        self.panelLayout.addWidget(self.canvasSplitter)

    def _check_for_highest_possible_comps(self, numSpectra: int) -> None:
        """
        It is not possible to solve for more principal components as there are spectra present.
        This method takes care for it.
        :param numSpectra:
        :return:
        """
        for spinbox in [self.pc1Selector, self.pc2Selector, self.pc3Selector]:
            if spinbox.value() > numSpectra:
                spinbox.valueChanged.disconnect()
                spinbox.setValue(numSpectra)
                spinbox.valueChanged.connect(self.update_all)

    def _update_cluster_plot(self) -> None:
        """
        Reads out data from specClusterer and refreshes the PCA Cluster Plot
        :return:
        """
        clusterMemberships = self.specClusterer.clusterMemberships
        numClusters: int = self.specClusterer.numDesiredClusters
        cntr = self.specClusterer.clusterCenters
        fpc = self.specClusterer.fpc

        if not self.pc3CheckBox.isChecked():
            self.pcAx2d.clear()
            xpts, ypts = self.specClusterer.xpts, self.specClusterer.ypts
            for j in range(numClusters):
                self.pcAx2d.plot(xpts[clusterMemberships == j], ypts[clusterMemberships == j], '.', picker=5)

            # mark the center of each cluster
            for pt in cntr:
                self.pcAx2d.plot(pt[0], pt[1], 'rs')
            self.pcAx2d.set_title(f'Centers = {numClusters}; fpc = {fpc:.3f}')
            self.pcAx2d.set_xlabel(f'PC {self.pc1Selector.value()} Scores')
            self.pcAx2d.set_ylabel(f'PC {self.pc2Selector.value()} Scores')
            self.pcaCanvas2d.draw()

            self.pcaCanvas2d.mpl_connect('pick_event', self.onpick)

        elif self.pc3CheckBox.isChecked():
            self.pcAx3d.clear()
            xpts, ypts, zpts = self.specClusterer.xpts, self.specClusterer.ypts, self.specClusterer.zpts
            for j in range(numClusters):
                self.pcAx3d.scatter(xpts[clusterMemberships == j], ypts[clusterMemberships == j],
                                    zpts[clusterMemberships == j], '.')

            # mark the center of each cluster
            for pt in cntr:
                self.pcAx3d.scatter(pt[0], pt[1], pt[2], 'rs')
            self.pcAx3d.set_title(f'Centers = {numClusters}; fpc = {fpc:.3f}')
            self.pcAx3d.set_xlabel(f'PC {self.pc1Selector.value()} Scores')
            self.pcAx3d.set_ylabel(f'PC {self.pc2Selector.value()} Scores')
            self.pcAx3d.set_zlabel(f'PC {self.pc3Selector.value()} Scores')
            self.pcaCanvas3d.draw()

    def _update_variance_plot(self) -> None:
        """
        The variance plot is updated with data from the specClusterer
        :return:
        """
        explVariance = self.specClusterer.explVariance
        self.var_ax.clear()
        self.var_ax.plot(np.arange(1, len(explVariance)+1), np.cumsum(explVariance))
        self.var_ax.set_xlabel('Number of Components')
        self.var_ax.set_ylabel('Cumulative Explained Variance (Ratio)')
        self.varCanvas.draw()

    def update_result_spectra(self):
        """
        The sorted spectra are retrieved from the specClusterer and sent to the sortedSpecViewer
        :return:
        """
        if self.specClusterer.sortedSpectra is not None:
            self.resultSpectraPlot.update_spectra(self.specClusterer.sortedSpectra)
            self.resultSpectraPlot.show()
        else:
            self.resultSpectraPlot.hide()

    # def _optimize_params(self) -> None:
    #     class ClusterCase:
    #         comp1: int
    #         comp2: int
    #         numClusters: int
    #         fpc: float
    #
    #     spectra: np.array = self.spectraContainer.get_selected_spectra()
    #     numSpectra: int = spectra.shape[1] - 1
    #     maxComponents: int = min(self.highestComponent, numSpectra)
    #     self.princComps, self.explVariance = fn.get_pca_of_spectra(spectra, numComponents=maxComponents)
    #     minClusters, maxClusters = 2, self.highestNumClusters
    #     results: list = []
    #     for comp1, comp2 in combinations(range(maxComponents), 2):
    #         for numClusters in np.arange(minClusters, maxClusters+1):
    #             cntr, cluster_membership, fpc = self._get_cluster_data(numClusters, comp1, comp2)
    #             result = ClusterCase()
    #             result.comp1 = comp1
    #             result.comp2 = comp2
    #             result.numClusters = numClusters
    #             result.fpc = fpc
    #             results.append(result)
    #
    #     allfpcs: list = [res.fpc for res in results]
    #     bestResult: ClusterCase = results[int(np.argmax(allfpcs))]
    #     self.pc1Selector.setValue(bestResult.comp1+1)
    #     self.pc2Selector.setValue(bestResult.comp2+1)
    #     self.numClusterSelector.setValue(bestResult.numClusters)
    #     QtWidgets.QMessageBox.about(self, 'Optimization done!',
    #                                 f'Best result with PC {bestResult.comp1+1} over PC {bestResult.comp2+1} and '
    #                                 f'{bestResult.numClusters} Clusters')

    def onpick(self, event: PickEvent) -> None:
        """
        matplotlib event for catching user inputs in 2d graph
        :param event:
        :return:
        """
        points: np.array = np.transpose(np.vstack((self.specClusterer.xpts, self.specClusterer.ypts)))
        point: np.array = np.transpose(np.array([event.mouseevent.xdata, event.mouseevent.ydata]))
        distances: np.array = np.linalg.norm(points-point, axis=1)
        pointIndex = np.argmin(distances)
        
        self.spectrumIndexSelected.emit(pointIndex)

    def closeEvent(self, event) -> None:
        self.resultSpectraPlot.close()
        event.accept()


class ResultSpectra(QtWidgets.QWidget):
    """
    GUI Object for showing the sorted and averaged spectra.
    """
    def __init__(self, mainWin, clusterWin):
        super(ResultSpectra, self).__init__()
        self.setWindowTitle('Resulting Spectra')

        self.mainWin = mainWin
        self.pcaClusterWin: PCAClusterView = clusterWin
        self.specClusterer: SpectraCluster = clusterWin.specClusterer
        self.refDB: ReferenceDatabase = ReferenceDatabase()
        self.refSelector: QtWidgets.QComboBox = QtWidgets.QComboBox()
        self.refSelector.currentIndexChanged.connect(self._update_references_in_plots)
        self.refSelector.setMinimumWidth(150)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        toolbar = QtWidgets.QGroupBox()
        toolbarLayout = QtWidgets.QHBoxLayout()
        toolbar.setLayout(toolbarLayout)

        updateSpecBtn = QtWidgets.QPushButton('Update Spectra from PCA Clustering')
        updateSpecBtn.released.connect(self.pcaClusterWin.update_result_spectra)

        setRefBtn = QtWidgets.QPushButton('Set Reference Spectra Directory')
        setRefBtn.released.connect(self._load_references)

        exportAvgBtn = QtWidgets.QPushButton('Export Averaged Spectra')
        exportAvgBtn.setMaximumWidth(200)
        exportAvgBtn.released.connect(self._export_average_spectra)

        self.numSpecPerClusterSpinbox = QtWidgets.QSpinBox()
        self.numSpecPerClusterSpinbox.setFixedWidth(50)
        self.numSpecPerClusterSpinbox.setMaximum(15)
        self.numSpecPerClusterSpinbox.setMinimum(1)
        self.numSpecPerClusterSpinbox.setValue(5)

        exportIndBtn = QtWidgets.QPushButton('Export Individual Spectra')
        exportIndBtn.setMaximumWidth(150)
        exportIndBtn.released.connect(self._export_individual_spectra)

        toolbarLayout.addWidget(updateSpecBtn)
        toolbarLayout.addWidget(setRefBtn)
        toolbarLayout.addWidget(self.refSelector)
        toolbarLayout.addStretch()
        toolbarLayout.addWidget(QtWidgets.QLabel('|'))
        toolbarLayout.addWidget(exportAvgBtn)
        toolbarLayout.addWidget(QtWidgets.QLabel('| Number of individual Spectra per Cluster to export:'))
        toolbarLayout.addWidget(self.numSpecPerClusterSpinbox)
        toolbarLayout.addWidget(exportIndBtn)

        splitter: QtWidgets.QSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        sortedSpectraGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox('Clustered Spectra')
        self.sortedSpectraLayout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        sortedSpectraGroup.setLayout(self.sortedSpectraLayout)

        self.sortedSpectraPlots: list = []
        self.averageSpecPlot: SpectraPlot = SpectraPlot(self.refDB)

        splitter.addWidget(sortedSpectraGroup)
        splitter.addWidget(self.averageSpecPlot)

        layout.addWidget(toolbar)
        layout.addWidget(splitter)

        self.sortedSpectraList: list = []
        self.averagedSpectra: list = []

    def _load_references(self) -> None:
        """
        Loads reference spectra from a user specified directory
        :return:
        """
        dirName = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Reference Spectra Folder', self.mainWin.dirPath)
        if dirName:
            self.refDB.load_refs_from_dir(dirName)
        self._update_reference_selector()

    def _update_reference_selector(self) -> None:
        """
        Reads the reference database object and updates the reference selector
        :return:
        """
        for index in range(self.refSelector.count()):
            self.refSelector.removeItem(index)
        self.refSelector.addItem('')
        self.refSelector.addItems(self.refDB.refNames)

    def _update_references_in_plots(self) -> None:
        """
        Updates the reference axis in all spectra plots
        :return:
        """
        specName: str = self.refSelector.currentText()
        spec: np.array = np.array([])
        if specName != '':
            index: int = self.refSelector.currentIndex() - 1
            spec = self.refDB.refSpectra[index]

        self.averageSpecPlot.update_reference_spectrum(specName, spec)
        for plot in self.sortedSpectraPlots:
            plot.update_reference_spectrum(specName, spec)

    def update_spectra(self, spectraList: list) -> None:
        """
        Retrieves a list of spectra arrays (1st Col: Wavenumbers, 2nd-nth Col, Intensities) per cluster
        and induces plot updates.
        :param spectraList:
        :return:
        """
        self.sortedSpectraList = spectraList
        numClusters: int = len(spectraList)
        self._clear_sorted_spectra()
        self._assert_having_n_sortedSpectraPlots(numClusters)
        self._set_canvas_width(numClusters)

        avgSpecLabels: list = []
        for index, spectra in enumerate(spectraList):
            specPlot: SpectraPlot = self.sortedSpectraPlots[index]
            specPlot.update_sample_spectra([spectra], f'{spectra.shape[1]} Spectra of cluster {index + 1}')
            self.sortedSpectraLayout.addWidget(specPlot)
            avgSpecLabels.append([f'Average of cluster {index+1}'])

        self._average_spectra(spectraList)
        self.averageSpecPlot.update_sample_spectra(self.averagedSpectra, 'Averaged Spectra', avgSpecLabels)

    def _clear_sorted_spectra(self) -> None:
        """
        The currently displayed plot-Widgetes are removed.
        :return:
        """
        for i in reversed(range(self.sortedSpectraLayout.count())):
            widget = self.sortedSpectraLayout.itemAt(i).widget()
            widget.setParent(None)
            self.sortedSpectraLayout.removeWidget(widget)

    def _assert_having_n_sortedSpectraPlots(self, n: int) -> None:
        """
        Checks, if sufficient spectra Plots are already present and creates more, if needed.
        :param n:
        :return:
        """
        if n > len(self.sortedSpectraPlots):
            for _ in range(n - len(self.sortedSpectraPlots)):
                newSpecPlot: SpectraPlot = SpectraPlot(self.refDB)
                self.sortedSpectraPlots.append(newSpecPlot)

    def _set_canvas_width(self, numCanvases: int) -> None:
        """
        Canvas Width is adjusted to retain constant relative dimensions, also when resizing the parent widget
        :param numCanvases:
        :return:
        """
        widgetWidthPx: int = int(round(self.width() / numCanvases))
        for specPlot in self.sortedSpectraPlots:
            specPlot.set_canvas_width(widgetWidthPx)

    def _average_spectra(self, sortedSpectra: np.array) -> None:
        """
        Takes an array of sorted spectra and performs averaging.
        :param sortedSpectra:
        :return:
        """
        self.averagedSpectra = []
        for index, spectra in enumerate(sortedSpectra):
            avgInt: np.array = np.mean(spectra[:, 1:], axis=1)
            avgSpec: np.array = np.transpose(np.vstack((spectra[:, 0], avgInt)))
            self.averagedSpectra.append(avgSpec)

    def _export_average_spectra(self) -> None:
        """
        The averaged spectra are written as csv to the source directory.
        :return:
        """
        for index, avgSpec in enumerate(self.averagedSpectra):
            self._save_spec_to_disk(avgSpec, f'Average of cluster {index + 1}')
        QtWidgets.QMessageBox.about(self, 'Export Finished', f'Successfully exported {len(self.averagedSpectra)} '
                                                             f'averaged spectra.')

    def _export_individual_spectra(self) -> None:
        """
        For each cluster, the desired number of individual spectra closest to the cluster centers are exported
        :return:
        """
        desiredSpecsPerCluster: int = self.numSpecPerClusterSpinbox.value()
        numClusters: int = len(self.specClusterer.clusterCenters)

        for clusterIndex in range(numClusters):
            specIndicesToExport: list = self._get_indices_of_spectra_closest_to_clusterIndex(desiredSpecsPerCluster,
                                                                                             clusterIndex)
            clusterSpectra = self.sortedSpectraList[clusterIndex]

            for runningIndex, specIndex in enumerate(specIndicesToExport):
                spec: np.array = clusterSpectra[:, [0, specIndex+1]]
                specName: str = f'Cluster {clusterIndex+1}, Spectrum {runningIndex+1}'
                self._save_spec_to_disk(spec, specName)

            numSpecToExport = len(specIndicesToExport)
            if numSpecToExport < desiredSpecsPerCluster:
                msg = f'Only {numSpecToExport} in cluster {clusterIndex+1}!\n' \
                      f'All these were export instead of the {desiredSpecsPerCluster} closest spectra per cluster.'
                QtWidgets.QMessageBox.warning(self, 'Warning', msg)

        QtWidgets.QMessageBox.about(self, 'Export finished',
                                    'Successfully exported individual spectra for all clusters.')

    def _get_indices_of_spectra_closest_to_clusterIndex(self, numSpectra: int, clusterIndex: int) -> list:
        """
        Calculates the indices of the numSpectra closest spectra to cluster of index clusterIndex
        :param numSpectra:
        :param clusterIndex:
        :return:
        """
        centers: np.array = self.specClusterer.clusterCenters
        numPrincComps: int = centers.shape[1]
        clusterMemberships: np.array = self.specClusterer.clusterMemberships
        pcaPoints: np.array = self.specClusterer.princComps[:, :numPrincComps]

        center = centers[clusterIndex]
        specIndicesOfCluster: np.array = np.where(clusterMemberships == clusterIndex)[0]
        pcaPointsOfCluster: np.array = pcaPoints[specIndicesOfCluster]
        numSpecInCluster: int = len(specIndicesOfCluster)
        distancesToCenter: np.array = np.linalg.norm(pcaPointsOfCluster - center, axis=1)
        sortedIndices = np.argsort(distancesToCenter)

        numSpecToExport = min([numSpecInCluster, numSpectra])
        specIndicesToExport: list = sortedIndices[:numSpecToExport]
        return specIndicesToExport

    def _save_spec_to_disk(self, spec: np.array, specName: str) -> None:
        """
        Writes the spectrum with the given Name to disk
        :param spec:
        :param specName:
        :return:
        """
        writeName: str = specName + '.csv'
        fname: str = os.path.join(self.mainWin.dirPath, writeName)
        np.savetxt(fname, spec, delimiter=',')
