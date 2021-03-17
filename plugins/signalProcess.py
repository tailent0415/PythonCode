import numpy as np
import scipy as sci
from scipy import signal
from scipy.linalg import lstsq


def custom_savgol_coeffs(window_length, polyorder, deriv=0, delta=1.0, pos=None, use="conv"):
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    halflen, rem = divmod(window_length, 2)

    if pos is None:
        pos = halflen

    if not (0 <= pos < window_length):
        raise ValueError("pos must be nonnegative and less than window_length.")

    if use not in ['conv', 'dot']:
        raise ValueError("use must be 'conv' or 'dot'")

    if deriv > polyorder:
        coeffs = np.zeros(window_length)
        return coeffs

    # interval value
    x = np.arange(-pos, window_length - pos, dtype=float)
    if use == "conv":
        x = x[::-1]

    # deriv level order
    order = np.arange(polyorder + 1).reshape(-1, 1)

    # matrix
    A = x ** order
    y = np.zeros(polyorder + 1)
    y[deriv] = sci.math.factorial(deriv) / (delta ** deriv)
    coeffs, _, _, _ = lstsq(A, y)
    return coeffs


# ================================Negative===================================#
# Negative of multiple#
def PltNeg(pltData=None):
    if pltData is None:
        return []
    pltName = type(pltData).__name__
    if pltName == 'priDisplay':
        for n, sig in enumerate(pltData.ScanSig):
            pltData.ScanSig[n].ydata = np.negative(sig.ydata)
    elif pltName == 'secDisplay':
        for n, sig in enumerate(pltData.IonSig):
            pltData.IonSig[n].ydata = np.negative(sig.ydata)
    else:
        raise TypeError("Source error")

    return pltData


# ================================FFT===================================#
# FFT of multiple#
def PltFFT(pltData=None):
    if pltData is None:
        return []
    pltName = type(pltData).__name__
    if pltName == 'priDisplay':
        for n, sig in enumerate(pltData.ScanSig):
            dataLen = len(sig.ydata)
            pltData.ScanSig[n].dt = 1 / sig.dt / dataLen
            pltData.ScanSig[n].ydata = np.fft.fft(sig.ydata).real
        pltData.tGraph.xlabel = 'Frequency (Hz)'
    elif pltName == 'secDisplay':
        for n, sig in enumerate(pltData.IonSig):
            dataLen = len(sig.ydata)
            pltData.IonSig[n].dt = 1 / sig.dt / dataLen
            pltData.IonSig[n].ydata = np.fft.fft(sig.ydata).real
        pltData.ionGraph.xlabel = 'Frequency (Hz)'
    else:
        raise TypeError("Source error")

    return pltData


# ================================Abs===================================#
# Absolute of multiple#
def PltAbs(pltData=None):
    if pltData is None:
        return []
    pltName = type(pltData).__name__
    if pltName == 'priDisplay':
        for n, sig in enumerate(pltData.ScanSig):
            pltData.ScanSig[n].ydata = np.absolute(sig.ydata)
    elif pltName == 'secDisplay':
        for n, sig in enumerate(pltData.IonSig):
            pltData.IonSig[n].ydata = np.absolute(sig.ydata)
    else:
        raise TypeError("Source error")

    return pltData


# ================================NumRectifier===================================#
# NumRectifier of single#
def NumRectifier(dt=-1, ydata=None):
    if dt <= 0:
        return []

    if ydata is None:
        return []

    C = 1e-12
    R = 1e+10

    kernelSize = 5
    kernel = np.zeros(kernelSize)
    halflen, rem = divmod(kernelSize, 2)

    kernel[0] = 1 * C / (12 * dt)
    kernel[1] = -2 * C / (3 * dt)
    kernel[2] = -1 / R
    kernel[3] = 2 * C / (3 * dt)
    kernel[4] = -1 * C / (12 * dt)
    kernel = kernel * 0.25

    I_b = sci.convolve(ydata, kernel, 'same')
    I_b[0:halflen] = I_b[halflen + 1]
    I_b[len(I_b) - halflen:len(I_b)] = I_b[len(I_b) - halflen - 1]

    return I_b


def NumRectifier_SG(dt=-1, ydata=None, winwidth=5):
    if ydata is None:
        return []

    if dt <= 0:
        return []

    if winwidth < dt:
        winwidth = dt * 5

    windowSize = int(winwidth / dt)
    if windowSize > 10000:
        windowSize = 10001

    halflen, rem = divmod(windowSize, 2)
    if rem == 0:
        windowSize = windowSize + 1

    d0 = custom_savgol_coeffs(window_length=windowSize, polyorder=3, deriv=0, delta=dt)
    d1 = custom_savgol_coeffs(window_length=windowSize, polyorder=3, deriv=1, delta=dt)

    k1 = 0.25
    C = 1e-12
    R = 1e+10

    v1 = sci.convolve(ydata, d0, 'same')
    dv1 = sci.convolve(ydata, d1, 'same')
    I_b = -1 * C * k1 * (dv1 + v1 / (C * R))

    I_b[0:halflen] = I_b[halflen + 1]
    I_b[len(I_b) - halflen:len(I_b)] = I_b[len(I_b) - halflen - 1]

    return I_b


# NumRectifier of multiple#
def PltNumRectifier(pltData=None):
    if pltData is None:
        pltData = []
    pltName = type(pltData).__name__
    if pltName == 'priDisplay':
        for n, sig in enumerate(pltData.ScanSig):
            pltData.ScanSig[n].ydata = NumRectifier(sig.dt, sig.ydata)
        pltData.tGraph.ylabel = 'Current (A)'
        pltData.mzGraph.ylabel = 'Current (A)'

    elif pltName == 'secDisplay':
        for n, sig in enumerate(pltData.IonSig):
            pltData.IonSig[n].ydata = NumRectifier(sig.dt, sig.ydata)
    else:
        raise TypeError("Source error")

    return pltData


def PltNumRectifier_SG(pltData=None, winwidth=5):
    if pltData is None:
        pltData = []
    pltName = type(pltData).__name__
    if pltName == 'priDisplay':
        for n, sig in enumerate(pltData.ScanSig):
            pltData.ScanSig[n].ydata = NumRectifier_SG(sig.dt, sig.ydata, winwidth)
        pltData.tGraph.ylabel = 'Current (A)'
        pltData.mzGraph.ylabel = 'Current (A)'

    elif pltName == 'secDisplay':
        for n, sig in enumerate(pltData.IonSig):
            pltData.IonSig[n].ydata = NumRectifier_SG(sig.dt, sig.ydata, winwidth)
    else:
        raise TypeError("Source error")

    return pltData


# ================================Filtering===================================#
# Filtering of single #
def Filtering(dt=-1, ydata=None, ktype='boxcar', kwidth=1):
    if dt <= 0 or ydata is None:
        return []

    if kwidth < dt:
        kwidth = dt
    elif kwidth > 3:
        kwidth = 3

    winsize = int(np.floor(kwidth / dt))
    if winsize != 1:
        if (ktype == 'norm' or ktype == 'Gaussian'):
            window = signal.windows.gaussian(winsize, winsize / 6)
        else:
            # boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen...
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window
            window = signal.get_window(ktype, winsize)

        window = window / sum(window)
        ydata = sci.convolve(ydata, window, 'same')
        halflen, rem = divmod(winsize, 2)
        if rem == 0:
            halflen = halflen + 1
        ydata[0:halflen] = ydata[halflen + 1]
        ydata[len(ydata) - halflen:len(ydata)] = ydata[len(ydata) - halflen - 1]

    return ydata


# Filtering of multiple#
def PltFiltering(pltData=None, ktype='boxcar', kwidth=1):
    if pltData is None:
        return []

    pltName = type(pltData).__name__
    if pltName == 'priDisplay':
        for n, sig in enumerate(pltData.ScanSig):
            pltData.ScanSig[n].ydata = Filtering(sig.dt, sig.ydata, ktype, kwidth)
    elif pltName == 'secDisplay':
        for n, sig in enumerate(pltData.IonSig):
            pltData.IonSig[n].ydata = Filtering(sig.dt, sig.ydata, ktype, kwidth)
    else:
        raise TypeError("Source error")

    return pltData


# ================================Convolution===================================#
# Convolution plot data#
def PltConv1D(pltData, kernel):
    halflen = int(len(kernel) / 2)
    pltName = type(pltData).__name__
    if pltName == 'priDisplay':
        for n, sig in enumerate(pltData.ScanSig):
            ydata = sci.convolve(sig.ydata, kernel, 'same')
            ydata[0:halflen] = ydata[halflen + 1]
            ydata[len(ydata) - halflen:len(ydata)] = ydata[len(ydata) - halflen - 1]
            pltData.ScanSig[n].ydata = ydata
    elif pltName == 'secDisplay':
        for n, sig in enumerate(pltData.IonSig):
            ydata = sci.convolve(sig.ydata, kernel, 'same')
            ydata[0:halflen] = ydata[halflen + 1]
            ydata[len(ydata) - halflen:len(ydata)] = ydata[len(ydata) - halflen - 1]
            pltData.IonSig[n].ydata = ydata
    else:
        raise TypeError("Source error")

    return pltData


# ===============================MakeKernel====================================#
# Maker kernel data#
def MakeKernel(dt=-1, ktype='boxcar', kwidth=float(1)):
    if dt <= 0:
        return []

    if kwidth < dt:
        kwidth = dt
    elif kwidth > 3:
        kwidth = 3

    items = int(np.floor(kwidth / dt))
    winsize = int(items / 2)

    if ktype == 'boxcar':
        kernelData = np.arange(winsize)
        kernelData = np.append(kernelData, [np.append(winsize, np.flip(kernelData))])
        kernelData = np.full(kernelData.shape[0], 1 / kernelData.shape[0])

    elif ktype == 'triang':
        kernelData = np.arange(winsize)
        kernelData = np.append(kernelData, [np.append(winsize, np.flip(kernelData))])
        kernelData = kernelData / np.sum(kernelData)

    elif ktype == 'norm' or ktype == 'Gaussian':
        w = winsize * (-1)
        vara = items / (items - 1)
        factor = np.zeros(items)
        alpha = (items / 6) ** 2 * 2

        for i in range(items):
            if i != 0:
                w = w + vara
            factor[i] = (w ** 2 * (-1)) / alpha

        expVal = np.exp(factor)
        kernelData = expVal / np.sum(expVal)

    else:
        raise ValueError("This kernel type isn't exist")

    return kernelData


# ==============================================================#
# =================Peak Finding Function========================#
# ==============================================================#

# Find peak processs
def FindPeak(peakData, pltData, butterWn=None, sdTimes=0, eleP=None):
    # butterWn is the butterworth filter cutoff frequency
    # sdTimes is wavelet removal range
    # eleP is "decline" or "rise"
    import scipy
    import pywt
    import graphConfigOfEditor as graphConfig
    from scipy import special, signal
    import matplotlib.pyplot as plt

    className = type(peakData).__name__
    if className != 'peakDisplay':
        raise TypeError("Peak Data Source error")
    className = type(pltData).__name__
    if className != 'priDisplay':
        raise TypeError("Plot Data Source error")

    if butterWn is None:
        halfCycle = 2 ** 8
        samples = len(pltData.ScanSig[0].ydata)
        butterWn = halfCycle / samples

    if eleP is None:  # Electric potential
        eleP = "decline"

    er = 150
    x = np.linspace(-1, 1, num=er)
    d = special.erf(x)
    d[(er - er // 6):] = 0
    d = d[::-1]
    d = (d + 1) * 0.5
    del er, x

    b, a = scipy.signal.butter(2, butterWn, btype='low', analog=False, output='ba')

    # db1-38 : daubechies
    # sym2-20 : symlets
    # morl : morlet
    # bior*.* : biorthogonal
    # coif1-17 : coiflets
    # mexh : mexican Hat
    # haar : Haar wavelet

    wavelet = pywt.Wavelet('haar')  # wavelet 分解方法
    waveletLevel = 3  # wavelet 層級
    peakInfo = []

    for n, sig in enumerate(pltData.ScanSig):

        bb, ba = scipy.signal.butter(2, sig.dt*10, btype='low', analog=False, output='ba')
        rectData = NumRectifier(sig.dt, sig.ydata)
        convData = signal.convolve(rectData, d, mode='valid', method='direct') / sum(d)
        filterData = scipy.signal.filtfilt(b, a, convData, method="gust")
        baseline = scipy.signal.filtfilt( bb, ba, convData, method="gust")

        plt.figure(10)
        plt.plot( filterData )
        plt.plot( baseline )

        plt.figure(1)
        plt.subplot(311)
        plt.title( "rectData" )
        plt.plot( rectData )

        plt.subplot(312)
        plt.title("convData")
        plt.plot( convData )

        plt.subplot(313)
        plt.title("filterData")
        plt.plot( filterData )

        coeffs = pywt.wavedec(filterData, wavelet=wavelet, level=waveletLevel)
        waveletData = []

        plt.figure(3)
        for idx in range(len(coeffs)):
            if idx == 0:
                waveletData = coeffs[0]
            else:
                coeffsArray = coeffs[idx]
                coeffsArray = np.pad(coeffsArray, (0, len(waveletData) - len(coeffs[idx])), 'constant',
                                     constant_values=(0, 0))
                if eleP == "decline":
                    tempArr = np.where(coeffsArray > 0, coeffsArray, 0)
                    rms = np.sqrt(np.mean(tempArr ** 2))
                    rms = np.sqrt(np.mean(np.where(tempArr > rms, 0, tempArr) ** 2))
                    sd = np.sqrt(np.mean((tempArr - rms) ** 2))
                    threshold = rms + sd * sdTimes
                    coeffsArray = pywt.threshold(coeffsArray, threshold, 'less')  # 向下邊線
                if eleP == "rise":
                    tempArr = np.where(coeffsArray < 0, coeffsArray, 0)
                    rms = np.sqrt(np.mean(tempArr ** 2))
                    rms = np.sqrt(np.mean(np.where( np.absolute(tempArr) > rms, 0, tempArr) ** 2))
                    sd = np.sqrt(np.mean((np.absolute(tempArr) - rms) ** 2))
                    threshold = rms + sd * sdTimes
                    coeffsArray = pywt.threshold(coeffsArray, threshold * -1, 'greater')  # 向上邊線
                plt.plot( coeffsArray )
                waveletData = pywt.idwt(waveletData, coeffsArray, wavelet=wavelet, mode='periodic')

        plt.figure(5)
        waveletData = waveletData[0:len(filterData)]
        newData = np.roll(np.roll(waveletData, -1) - waveletData, 1)
        newData[0] = 0
        zeroIdx = np.asarray(np.where(newData == 0))[0]
        items = 0
        tagX = []
        tagY = []
        endPoint = 0
        startPoint = 0
        waveletData = waveletData-baseline
        rms = np.sqrt(np.mean(waveletData ** 2))
        for idx, val in enumerate(zeroIdx):
            if startPoint < endPoint or idx == 0:
                startPoint = val
                continue
            else:
                if zeroIdx[idx - 1] == val - 1 or zeroIdx[idx - 1] == val - 2:
                    items = items + (zeroIdx[idx] - zeroIdx[idx - 1])
                    getPoint = False
                else:
                    getPoint = True
            if len(zeroIdx) - 1 == idx:
                getPoint = True
            if getPoint:
                if items > 0:
                    if startPoint != 0:
                        try:
                            if eleP == "decline":
                                while filterData[startPoint + items + 1] - filterData[startPoint + items] <= 0:
                                    items += 1
                            if eleP == "rise":
                                while filterData[startPoint + items + 1] - filterData[startPoint + items] >= 0:
                                    items += 1
                        except Exception as e:
                            break
                        endPoint = startPoint + items
                        print( waveletData[endPoint] )
                        if np.absolute( waveletData[endPoint] ) > rms:
                            tagX.append(startPoint)
                            tagY.append(items)

                startPoint = val
                items = 0

        # plt.figure(11)
        kernelHalfLen = len(d) // 2

        # x = np.linspace(0, len(newData) - 1, len(newData))
        tagIdx = []
        for idx, val in enumerate(tagX):
            tagIdx.append(val + tagY[idx] + kernelHalfLen)
            # plt.plot(x[tagIdx], peakData[tagIdx], 'bo')
            # plt.text(x[tagIdx], peakData[tagIdx], tagY[idx], fontsize=12)

        newData = np.hstack((np.full_like(np.arange(kernelHalfLen), filterData[kernelHalfLen], dtype=np.double),
                             filterData,
                             np.full_like(np.arange(kernelHalfLen), filterData[(kernelHalfLen + 1) * -1],
                                          dtype=np.double)))

        if len(newData) > len(pltData.ScanSig[n].ydata):
            newData = np.delete(newData, [-1], None)

        pltData.ScanSig[n].ydata = newData
        peakInfo.append(([], tagIdx, [], [], tagY))

        plt.figure(2)
        plt.plot( pltData.ScanSig[n].xdata, pltData.ScanSig[n].ydata )

        newX = []
        newY = []
        print( tagIdx )
        for idx, val in enumerate( tagIdx ):
            newX.append( pltData.ScanSig[n].xdata[val] )
            newY.append( pltData.ScanSig[n].ydata[val])
        plt.plot( newX, newY, 'r*' )

    peakData = graphConfig.init('peak', peakInfo)
    del rectData, convData, filterData, baseline

    '''
    # 小波包使用方法
    wp = pywt.WaveletPacket(data=filterData, wavelet=wavelet, mode='periodic', maxlevel=waveletLevel)
    for idx, node in enumerate(wp.get_level(waveletLevel, 'freq')):
        figIdx = idx // 4 + 4
        plt.figure(figIdx)
        if idx % 4 == 0:
            plt.subplot(221)
        if idx % 4 == 1:
            plt.subplot(222)
        if idx % 4 == 2:
            plt.subplot(223)
        if idx % 4 == 3:
            plt.subplot(224)
        #if idx > 0:
            #wp[node.path].data = np.zeros_like(wp[node.path].data)
        plt.title(node.path)
        plt.plot(wp[node.path].data)
    plt.figure(30)
    plt.plot(wp.reconstruct(update=True))
    '''

    return peakData, pltData


# Get remove baseline Y data #
def GetBaseline(xdata=None, ydata=None, iteration=30):
    if ydata is None or xdata is None:
        return []

    threshold = np.sum(np.absolute(ydata)) * 1e-10
    weight = np.ones(len(ydata))
    condition = True
    idx = 0
    baseline = []
    newY = ydata
    while condition:
        poly = np.polyfit(xdata, ydata, 2, w=weight)
        baseline = np.polyval(poly, xdata)
        newY = ydata - baseline
        maxVal = float("-inf")
        sumVal = 0
        for n, val in enumerate(newY):
            if val < 0:
                sumVal = sumVal + val
                if val > maxVal:
                    maxVal = val
        sumVal = np.absolute(sumVal)
        if sumVal < threshold:
            break
        for n, val in enumerate(newY):
            if val >= 0:
                weight[n] = 0
            else:
                weight[n] = np.exp(np.absolute(((idx + 1) * val) / sumVal))

        weight[n] = maxVal / sumVal
        weight[len(weight) - 1] = maxVal / sumVal

        if idx > iteration:
            condition = False
        idx = idx + 1

    poly = np.polyfit(xdata, newY, 3)
    baseline = np.polyval(poly, xdata)
    ydata = newY - baseline
    return ydata


# Get peak index and X value #
def GetPeak(dt=None, ydata=None, winwidth=3, ampTh=float("-inf"), slopeTh=float("-inf")):
    if dt is None or ydata is None:
        return []

    derive1 = np.gradient(ydata, dt)
    derive1[0] = derive1[1]
    derive1[derive1.size - 1] = derive1[derive1.size - 2]
    lastVal = 0
    peakIdx = []
    for n, val in enumerate(derive1):
        if winwidth < n < derive1.size - winwidth:
            if lastVal > 0 and val <= 0:
                if ydata[n - 1] > ydata[n]:
                    dataIdx = n - 1
                else:
                    dataIdx = n
                if ydata[dataIdx] > ampTh and abs(derive1[n]) > slopeTh:
                    peakIdx.append(dataIdx)
        lastVal = val
    return peakIdx


# Get peak left and right point information #
def GetPeakWidth(dt=None, ydata=None, peakIdx=None):
    if dt is None or ydata is None or peakIdx is None:
        return [], [], [], []

    derive1 = np.gradient(ydata, dt)
    lastVal = 0
    tagetIdx = 0
    pervIdx = 0

    leftIdx = []
    rightIdx = []

    for n, val in enumerate(derive1):
        if tagetIdx >= len(peakIdx):
            break
        if lastVal <= 0 and val > 0:
            if n > peakIdx[tagetIdx]:
                leftIdx.append(pervIdx)
                rightIdx.append(n)
                tagetIdx = tagetIdx + 1
                pervIdx = n
            else:
                pervIdx = n
        lastVal = val

    if tagetIdx < len(peakIdx):
        leftIdx.append(pervIdx)
        rightIdx.append(n)

    return leftIdx, rightIdx


# Get peak half height #     
def GetPeakHalfHeight(xdata=[], ydata=[], peakIdx=[], leftIdx=[], rightIdx=[]):
    if len(ydata) == 0 or len(xdata) == 0:
        return []
    halfHeight = []
    for n, val in enumerate(peakIdx):
        xp = [xdata[leftIdx[n]], xdata[rightIdx[n]]]
        fp = [ydata[leftIdx[n]], ydata[rightIdx[n]]]
        halfHeight.append((ydata[peakIdx[n]] - np.interp(xdata[peakIdx[n]], xp, fp)) * 0.5)
    return halfHeight
