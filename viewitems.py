from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

import functions as fn


class SpectrumView(QtWidgets.QGraphicsView):
    def __init__(self, parentContainer, spectrum: np.array = None, specIndex: int = None):
        super(SpectrumView, self).__init__()
        self.parentContainer = parentContainer
        self.spectrum = spectrum
        self.spectrumToPlot = None
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
        # self.update_specGraph()

    def update_spectra_options(self, subtractBaseline: bool, removeCO2: bool):
        if self.spectrum is not None:
            self.spectrumToPlot = self.spectrum.copy()
            if subtractBaseline:
                self.spectrumToPlot[:, 1] -= fn.get_baseline(self.spectrumToPlot[:, 1], smoothness_param=1e6)

    def update_specGraph(self):
        if self.spectrumToPlot is None:
            newPixmap: QtGui.QPixmap = self._spectrum_to_pixmap(self.spectrum)
        else:
            newPixmap: QtGui.QPixmap = self._spectrum_to_pixmap(self.spectrumToPlot)
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

        # sortedSpectra: list = sort_spectra(self.spectraContainer.get_selected_spectra(),
        # cluster_membership, numClusters)
        # self.resultSpectraPlot.update_spectra(sortedSpectra)

    def closeEvent(self, event) -> None:
        self.resultSpectraPlot.close()
        event.accept()


class ResultSpectra(QtWidgets.QWidget):
    def __init__(self):
        super(ResultSpectra, self).__init__()
        self.setWindowTitle('Resulting Spectra')
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.sortedSpectraCanvas = FigureCanvas(Figure())
        self.averageSpectraCanvas = FigureCanvas(Figure())
        splitter.addWidget(self.sortedSpectraCanvas)
        splitter.addWidget(self.averageSpectraCanvas)
        layout.addWidget(splitter)

    def update_spectra(self, spectraList: list) -> None:
        # spectraList is nested list of spectra per cluster...
        pass


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = SpectrumView()
    win.show()
    ret = app.exec_()