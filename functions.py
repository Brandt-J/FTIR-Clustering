# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:37:04 2020

@author: xbrjos
"""
import os
import numpy as np
import csv
from scipy.linalg import solveh_banded
import matplotlib.pyplot as plt


def read_format_and_save_spectra(basedir: str, allSpecFiles: list):
    """
    Reads each file in list of Files. A numpy binary file is created with all spectra after having read all csv files.
    This makes later import faster.
    :param basedir:
    :param allSpecFiles:
    :return:
    """
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


def get_noise_level(intensities: np.array, axis=0, ddof=0):
    """
    Takes a one-dimensional intensities array and estimates an arbitraty noise level.
    Larger numbers usually represent more noisy spectra, but this is not always the case.
    A more sophistcated signal-to-noise-algorithm would be useful here.
    :param intensities:
    :param axis:
    :param ddof:
    :return:
    """
    mean = np.mean(intensities)
    stdev = np.std(intensities)
    return mean/stdev


def get_baseline(intensities, asymmetry_param=0.05, smoothness_param=1e4,
                 max_iters=5, conv_thresh=1e-5, verbose=False):
    """Computes the asymmetric least squares baseline.
    * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
    smoothness_param: Relative importance of smoothness of the predicted response.
    asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                         Setting p=1 is effectively a hinge loss.
    """
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


def remove_co2(spectrum: np.array) -> np.array:
    """
    Removes the CO2 region of a spectrum.
    :param spectrum:
    :return:
    """
    def findIndexClosestToValue(data: np.array, value: float) -> int:
        diff: np.array = abs(data - value)
        minDiff: float = np.min(diff)
        return int(np.where(diff == minDiff)[0])

    startCO2: int = findIndexClosestToValue(spectrum[:, 0], 2100)
    endCO2: int = findIndexClosestToValue(spectrum[:, 0], 2400)

    numValuesToOverride: int = int(endCO2 - startCO2)
    startVal, endVal = spectrum[startCO2, 1], spectrum[endCO2, 1]

    newYData: np.array = np.linspace(startVal, endVal, numValuesToOverride)
    spectrum[startCO2:endCO2, 1] = newYData
    return spectrum
