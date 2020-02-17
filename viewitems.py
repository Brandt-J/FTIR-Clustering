from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

import functions as fn


class SpectrumView(QtWidgets.QGraphicsView):
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

        scene = QtWidgets.QGraphicsScene(self)
        scene.setItemIndexMethod(QtWidgets.QGraphicsScene.NoIndex)
        scene.setBackgroundBrush(QtCore.Qt.darkGray)
        self.setScene(scene)
        self.setCacheMode(QtWidgets.QGraphicsView.CacheBackground)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.BoundingRectViewportUpdate)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)

        self.item = QtWidgets.QGraphicsPixmapItem()
        self.item.setPos(0, 0)

        self.scene().addItem(self.item)
        self.item.setScale(0.45)

        self.isSelected: bool = True
        self._do_auto_deselection()

    def update_spectra_options(self, subtractBaseline: bool, removeCO2: bool):
        if self.origSpectrum is not None:
            self.spectrum = self.origSpectrum.copy()
            if subtractBaseline:
                self.spectrum[:, 1] -= fn.get_baseline(self.spectrum[:, 1], smoothness_param=1e6)
            if removeCO2:
                self.spectrum = fn.remove_co2(self.spectrum)
        else:
            self.spectrum = None

    def update_specGraph(self):
        newPixmap: QtGui.QPixmap = self._spectrum_to_pixmap(self.spectrum)
        self.item.setPixmap(newPixmap)

    def _do_auto_deselection(self) -> None:
        if self.spectrum is None:
            self.isSelected = False
        else:
            noiseLevel: float = fn.get_noise_level(self.spectrum[:, 1])
            if noiseLevel > 3:
                self.isSelected = False
        self._update_opacity()

    def _spectrum_to_pixmap(self, spectrum: np.array) -> QtGui.QPixmap:
        # sp = SubplotParams(left=0., bottom=0., right=1., top=1.)
        fig = Figure()
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

    def _update_opacity(self) -> None:
        if self.isSelected:
            self.item.setOpacity(1.)
        else:
            self.item.setOpacity(0.2)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MiddleButton:
            self.drag = event.pos()

        elif event.button() == QtCore.Qt.LeftButton:
            self.isSelected = not self.isSelected
            self._update_opacity()
            self.parentContainer.update_spec_selection()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.drag is not None:
            p0 = event.pos()
            move = self.drag - p0
            self.horizontalScrollBar().setValue(move.x() + self.horizontalScrollBar().value())
            self.verticalScrollBar().setValue(move.y() + self.verticalScrollBar().value())

            self.drag = p0

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self.drag = None

    def wheelEvent(self, event):
        factor: float = 1.01 ** (event.angleDelta().y() / 8)
        self.item.setScale(self.item.scale() * factor)


class PCAClusterView(QtWidgets.QWidget):
    def __init__(self, spectraContainer):
        super(PCAClusterView, self).__init__()

        self.spectraContainer = spectraContainer
        self.princComps = None

        self.setWindowTitle('PCA Clustering')
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        toolbar = QtWidgets.QGroupBox('Clustering Controls')
        toolbarLayout = QtWidgets.QHBoxLayout()
        toolbar.setLayout(toolbarLayout)
        toolbar.setMaximumHeight(100)

        self.pc1Selector = QtWidgets.QSpinBox()
        self.pc2Selector = QtWidgets.QSpinBox()
        for index, spinbox in enumerate([self.pc1Selector, self.pc2Selector], 1):
            spinbox.setMinimum(1)
            spinbox.setMaximum(10)
            spinbox.setValue(index)
            spinbox.setFixedWidth(50)
            spinbox.valueChanged.connect(self._update_pca)

        self.numClusterSelector = QtWidgets.QSpinBox()
        self.numClusterSelector.setMaximum(1)
        self.numClusterSelector.setMaximum(7)
        self.numClusterSelector.setValue(3)
        self.numClusterSelector.setFixedWidth(70)
        self.numClusterSelector.valueChanged.connect(self._update_clustering)

        toolbarLayout.addWidget(QtWidgets.QLabel('Plot PC'))
        toolbarLayout.addWidget(self.pc1Selector)
        toolbarLayout.addWidget(QtWidgets.QLabel(' over'))
        toolbarLayout.addWidget(self.pc2Selector)

        toolbarLayout.addWidget(QtWidgets.QLabel(' | Num. Clusters in PCA:'))
        toolbarLayout.addWidget(self.numClusterSelector)

        updateSpecsBtn = QtWidgets.QPushButton('Update Spectra View')
        updateSpecsBtn.released.connect(self.update_result_spectra)

        toolbarLayout.addWidget(QtWidgets.QLabel(' |'))
        toolbarLayout.addWidget(updateSpecsBtn)
        toolbarLayout.addStretch()

        layout.addWidget(toolbar)

        self.pcaCanvas = FigureCanvas(Figure())
        self.pc_ax = self.pcaCanvas.figure.add_subplot(111)
        layout.addWidget(self.pcaCanvas)
        self.pcaCanvas.draw()

        self.resultSpectraPlot = ResultSpectra()

    def update_all(self) -> None:
        self._update_pca()
        self._update_clustering()

    def _update_pca(self) -> None:
        spectra: np.array = self.spectraContainer.get_selected_spectra()
        maxComp: int = max([self.pc1Selector.value(), self.pc2Selector.value()])
        self.princComps = fn.get_pca_of_spectra(spectra, numComponents=maxComp)
        self._update_clustering()

    def _update_clustering(self) -> None:
        self.pc_ax.clear()
        comp1Ind = self.pc1Selector.value() - 1
        comp2Ind = self.pc2Selector.value() - 1
        xpts = self.princComps[:, comp1Ind]
        ypts = self.princComps[:, comp2Ind]
        numClusters = self.numClusterSelector.value()

        cntr, cluster_membership, fpc = fn.cluster_data(xpts, ypts, numClusters)
        for j in range(numClusters):
            self.pc_ax.plot(xpts[cluster_membership == j], ypts[cluster_membership == j], '.')

        # Mark the center of each cluster
        for pt in cntr:
            self.pc_ax.plot(pt[0], pt[1], 'rs')

        self.pc_ax.set_title(f'Centers = {numClusters}; fpc = {fpc:.3f}')
        self.pcaCanvas.draw()

        self.sortedSpectra: list = fn.sort_spectra(self.spectraContainer.get_selected_spectra(),
                                                   cluster_membership, numClusters)

    def update_result_spectra(self):
        if self.sortedSpectra is not None:
            self.resultSpectraPlot.update_spectra(self.sortedSpectra)
            self.resultSpectraPlot.show()

    def closeEvent(self, event) -> None:
        self.resultSpectraPlot.close()
        event.accept()


class ResultSpectra(QtWidgets.QWidget):
    def __init__(self):
        super(ResultSpectra, self).__init__()
        self.setWindowTitle('Resulting Spectra')
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        splitter: QtWidgets.QSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        sortedSpectraGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox('Clustered Spectra')
        self.sortedSpectraLayout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        sortedSpectraGroup.setLayout(self.sortedSpectraLayout)

        self.sortedSpectraCanvases: list = []
        self.sortedSpectraAxes: list = []
        self.averageSpectraCanvas: FigureCanvas = FigureCanvas(Figure())
        self.averageAx = self.averageSpectraCanvas.figure.add_subplot(111)

        splitter.addWidget(sortedSpectraGroup)
        splitter.addWidget(self.averageSpectraCanvas)
        layout.addWidget(splitter)

    def update_spectra(self, spectraList: list) -> None:
        # spectraList is nested list of spectra per cluster...
        numClusters: int = len(spectraList)
        self._clear_sorted_spectra()
        self._assert_having_n_canvases(numClusters)
        self._set_canvas_width(numClusters)

        for index, spectra in enumerate(spectraList):
            self._plot_sorted_spectra(spectra, index)
            self._add_canvas_to_layout(index)

        self._plot_averaged_spectra(spectraList)

    def _clear_sorted_spectra(self) -> None:
        for i in reversed(range(self.sortedSpectraLayout.count())):
            widget = self.sortedSpectraLayout.itemAt(i).widget()
            widget.setParent(None)
            self.sortedSpectraLayout.removeWidget(widget)

    def _assert_having_n_canvases(self, n: int):
        if n > len(self.sortedSpectraCanvases):
            for _ in range(n - len(self.sortedSpectraCanvases)):
                newCanvas: FigureCanvas = FigureCanvas(Figure())
                self.sortedSpectraCanvases.append(newCanvas)
                newAx = newCanvas.figure.add_subplot(111)
                self.sortedSpectraAxes.append(newAx)

    def _set_canvas_width(self, numCanvases: int) -> None:
        widgetWidthMM: int = self.widthMM()
        widgetWidthInch: float = widgetWidthMM / 25.4
        widthPerCanvas: float = widgetWidthInch / numCanvases
        for canvas in self.sortedSpectraCanvases:
            fig: Figure = canvas.figure
            fig.set_figwidth(widthPerCanvas)

    def _plot_sorted_spectra(self, spectra: np.array, index: int):
        ax = self.sortedSpectraAxes[index]
        ax.clear()
        for specInd in range(spectra.shape[1] - 1):
            ax.plot(spectra[:, 0], spectra[:, specInd + 1])
        ax.set_title(f'{spectra.shape[1]-1} Spectra of cluster {index + 1}')
        ax.set_xlabel('Wavenumber (cm-1)')
        ax.set_ylabel('Abundancy (a.u.)')

    def _add_canvas_to_layout(self, index):
        canvas: FigureCanvas = self.sortedSpectraCanvases[index]
        self.sortedSpectraLayout.addWidget(canvas)
        canvas.draw()

    def _plot_averaged_spectra(self, sortedSpectra: np.array):
        self.averageAx.clear()
        for index, spectra in enumerate(sortedSpectra):
            avgInt: np.array = np.mean(spectra[:, 1:], axis=1)
            self.averageAx.plot(spectra[:, 0], avgInt, label=f'Average of Cluster {index + 1}')
        self.averageAx.set_title('Averaged Spectra')
        self.averageAx.set_xlabel('Wavenumber (cm-1)')
        self.averageAx.set_ylabel('Abundancy (a.u.)')
        self.averageAx.legend()
        self.averageSpectraCanvas.draw()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = SpectrumView()
    win.show()
    ret = app.exec_()