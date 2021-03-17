# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 21:24:02 2018
Last modified on Thu Aug 30 23:24:05 2018

@author: S CHOU

These functions extract information from the .MAT files exported by ATI QITExplorer.exe
"""
import scipy.io as sio
import numpy as np


class msLoadMAT():
    def __init__(self, filepath):
        self.mat = sio.loadmat(filepath)

    def get_ionsig(self):
        self.ionsigs = []
        i = 0
        for key in self.mat.keys():
            if key[:11] == 'ionisationY':
                self.ionsigs.append({'dt': self.mat['ionisation_dt_' + str(i)][0][0],
                                     'ydata': self.mat['ionisationY_' + str(i)].flatten()})
                i = i + 1
        return self.ionsigs

    def get_spectrum(self):
        self.spectrum = []
        i = 0
        for key in self.mat.keys():
            if key[:9] == 'spectrumX':
                self.spectrum.append({'dt': self.mat['spectrum_dt_' + str(i)][0][0],
                                      'xdata': self.mat['spectrumX_' + str(i)].flatten(),
                                      'ydata': self.mat['spectrumY_' + str(i)].flatten()})
                i = i + 1
        return self.spectrum

    def get_starttime(self):
        return self.mat['signal_t0'][0][0]


class wfmLoadMAT():
    def __init__(self, filepath):
        self.mat = sio.loadmat(filepath)
        self.ais = []
        for key in self.mat.keys():
            if "ai" in key:
                self.ais.append(key)

    def channels(self):
        return self.ais

    def y(self, key):  # Return the waveform data of a specific channel
        return self.mat[key].flatten()

    def yt(self, key):
        return self.mat[key].flatten()

    def dt(self):
        return self.mat['dt'][0][0]

    def time(self):
        return self.mat['t0'][0][0]


class wfmSaveMAT():
    def __init__(self):
        self.mat = {'dt': 1, 't0': 0}

    def addwfm(self, key, yy):
        self.mat.update({key: yy})

    def dt(self, tt):
        self.mat['dt'] = tt

    def time(self, tt):
        self.mat['t0'] = tt

    def savewfms(self, folder, filename):
        filepath = folder + "\\" + filename
        sio.savemat(filepath, self.mat)