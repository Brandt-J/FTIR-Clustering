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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.linalg import solveh_banded
import skfuzzy
import numpy as np


class SpectraCluster(object):
    """
    Dedicated object for performing PCA, fuzzy-clustering and sorting of spectra.
    """
    def __init__(self, spectraContainer):
        self.spectraContainer = spectraContainer
        self.spectra: np.ndarray= None
        self.numComponents = None
        self.princComps, self.explVariance = None, None

        self.highestComponent = 15
        self.highestNumClusters = 5

        self.numDesiredClusters = None
        self.clusterComponents: tuple = None, None, None

        self.xpts, self.ypts, self.zpts = None, None, None
        self.clusterMemberships = None
        self.clusterCenters = None
        self.fpc = None

        self.sortedSpectra = []

    def update(self) -> None:
        """
        Recomputes all data.
        :return:
        """
        assert self.spectra is not None
        assert self.clusterComponents is not None
        assert self.numDesiredClusters is not None
        self._update_pca()
        self._cluster_spectra()
        self._sort_spectra()

    def _update_pca(self) -> None:
        """
        Reruns the PCA.
        :return:
        """
        numSpectra: int = self.spectra.shape[1] - 1
        maxComponents = min(numSpectra, self.highestComponent)
        intensities = self.spectra[:, 1:]
        intensitiesStandardized = StandardScaler().fit_transform(intensities)
        pca = PCA(n_components=maxComponents)
        self.princComps = pca.fit_transform(np.transpose(intensitiesStandardized))
        self.explVariance = pca.explained_variance_ratio_

    def _cluster_spectra(self) -> None:
        """
        Takes the principal Components and runs fuzzy clustering algorithm.
        :return:
        """
        alldata = np.array([])
        comp1Ind, comp2Ind, comp3Ind = self.clusterComponents
        self.xpts = self.princComps[:, comp1Ind]
        self.ypts = self.princComps[:, comp2Ind]
        self.zpts = None
        if comp3Ind is not None:
            self.zpts = self.princComps[:, comp3Ind]

        if self.zpts is None:
            alldata = np.vstack((self.xpts, self.ypts))
        else:
            alldata = np.vstack((self.xpts, self.ypts, self.zpts))
        self.clusterCenters, u, u0, d, jm, p, self.fpc = skfuzzy.cluster.cmeans(alldata, self.numDesiredClusters, 2,
                                                                                error=0.005, maxiter=1000, init=None)
        self.clusterMemberships = np.argmax(u, axis=0)

    def _sort_spectra(self) -> None:
        """
        Sorts the spectra according to their cluster membership.
        :return:
        """
        self.sortedSpectra = []
        for clusterIndex in range(self.numDesiredClusters):
            specIndices = np.where(self.clusterMemberships == clusterIndex)[0]
            # increment by one and insert 0, as we also need to copy the wavenumbers
            specIndices += 1
            specIndices = np.insert(specIndices, 0, 0)
            self.sortedSpectra.append(self.spectra[:, specIndices])
