# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:37:04 2020

@author: xbrjos
"""
import os
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.linalg import solveh_banded
import skfuzzy
import matplotlib.pyplot as plt


def read_format_and_save_spectra(basedir, allSpecFiles):
    wavenumbers = []
    spectra = []
    
    for specFile in allSpecFiles:
        curWavenumbers = []
        curSpec = []
        with open(os.path.join(basedir, specFile)) as fp:
            lines = csv.reader(fp, delimiter=',')
            for line in lines:
                curWavenumbers.append(float(line[0]))
                curSpec.append(float(line[1]))
        wavenumbers.append(curWavenumbers)
        spectra.append(curSpec)

    for index, wavenumber in enumerate(wavenumbers[0]):
        allWavenumbersIdentical = True
        for i in range(len(wavenumbers)):
            if wavenumbers[i][index] != wavenumber:
                allWavenumbersIdentical = False
                print('non identical wavenumber:', wavenumber)

    assert allWavenumbersIdentical
            
    allSpectra = [wavenumbers[0]]
    for spec in spectra:
        allSpectra.append(spec)
    print('num of spectra + wavenumbers', len(allSpecFiles))

    allSpectra = np.array(allSpectra)
    allSpectra = np.transpose(allSpectra)
    np.save(os.path.join(basedir, 'allSpectra.npy'), allSpectra)
    print('successfully created npy file')
    return allSpectra


def get_pca_of_spectra(spectra: np.array, numComponents: int = 2) -> np.array:
    intensities = spectra[:, 1:]
    intensitiesStandardized = StandardScaler().fit_transform(intensities)
    pca = PCA(n_components=numComponents)
    return pca.fit_transform(np.transpose(intensitiesStandardized))


def cluster_data(xpts, ypts, numOfClusters):
    alldata = np.vstack((xpts, ypts))
    cntr, u, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans(alldata, numOfClusters, 2, error=0.005, maxiter=1000, init=None)
    cluster_membership = np.argmax(u, axis=0)
    return cntr, cluster_membership, fpc


def sort_spectra(spectra, clusterAssignments, numberOfClusters) -> list:
    sortedSpectra = []
    for clusterIndex in range(numberOfClusters):
        specIndices = np.where(clusterAssignments == clusterIndex)[0]
        # increment by one and insert 0, as we also need to copy the wavenumbers
        specIndices += 1
        specIndices = np.insert(specIndices, 0, 0)
        sortedSpectra.append(spectra[:, specIndices])
    return sortedSpectra


def get_noise_level(intensities: np.array, axis=0, ddof=0):
    mean = np.mean(intensities)
    stdev = np.std(intensities)
    return mean/stdev


def get_baseline(intensities, asymmetry_param=0.05, smoothness_param=1e4,
                 max_iters=5, conv_thresh=1e-5, verbose=False):
    '''Computes the asymmetric least squares baseline.
    * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
    smoothness_param: Relative importance of smoothness of the predicted response.
    asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                         Setting p=1 is effectively a hinge loss.
    '''
    smoother = WhittakerSmoother(intensities, smoothness_param, deriv_order=2)
    # Rename p for concision.
    p = asymmetry_param
    # Initialize weights.
    w = np.ones(intensities.shape[0])
    baseline = None
    for i in range(max_iters):
        baseline = smoother.smooth(w)
        mask = intensities > baseline
        new_w = p * mask + (1 - p) * (~mask)
        conv = np.linalg.norm(new_w - w)
        if verbose:
            print(i + 1, conv)
        if conv < conv_thresh:
            break
        w = new_w
    # else:
    #     print('ALS did not converge in %d iterations' % max_iters)
    return baseline


class WhittakerSmoother(object):
    def __init__(self, signal, smoothness_param, deriv_order=1):
        self.y = signal
        assert deriv_order > 0, 'deriv_order must be an int > 0'
        # Compute the fixed derivative of identity (D).
        d = np.zeros(deriv_order * 2 + 1, dtype=int)
        d[deriv_order] = 1
        d = np.diff(d, n=deriv_order)
        n = self.y.shape[0]
        k = len(d)
        s = float(smoothness_param)

        # Here be dragons: essentially we're faking a big banded matrix D,
        # doing s * D.T.dot(D) with it, then taking the upper triangular bands.
        diag_sums = np.vstack([
            np.pad(s * np.cumsum(d[-i:] * d[:i]), ((k - i, 0),), 'constant')
            for i in range(1, k + 1)])
        upper_bands = np.tile(diag_sums[:, -1:], n)
        upper_bands[:, :k] = diag_sums
        for i, ds in enumerate(diag_sums):
            upper_bands[i, -i - 1:] = ds[::-1][:i + 1]
        self.upper_bands = upper_bands

    def smooth(self, w):
        foo = self.upper_bands.copy()
        foo[-1] += w  # last row is the diagonal
        return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)
