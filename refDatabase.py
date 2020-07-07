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
import os
import csv
import numpy as np
from PyQt5 import QtWidgets


class ReferenceDatabase(object):
    def __init__(self):
        super(ReferenceDatabase, self).__init__()
        self.refSpectra: list = []
        self.refNames: list = []

    def load_refs_from_dir(self, dirName: str) -> None:
        """
        Loads all files from a given directory
        :param dirName:
        :return:
        """
        self.refSpectra = []
        self.refNames = []
        fnames = os.listdir(dirName)
        for fname in fnames:
            wavenumbers: list = []
            intensities: list = []
            with open(os.path.join(dirName, fname)) as fp:
                lines = csv.reader(fp, delimiter=';')
                for line in lines:
                    wavenumber: float = float(line[0].replace(',', '.'))
                    intensity: float = float(line[1].replace(',', '.'))
                    wavenumbers.append(wavenumber)
                    intensities.append(intensity)
            
            self.refNames.append(fname.split('.')[0])
            spec: np.ndarray= np.transpose(np.vstack((wavenumbers, intensities)))
            self.refSpectra.append(spec)
