def init(sigType, data=None):
    if data is None:
        data = []
    if sigType == 'pri':
        struct = priDisplay( data )
    else:
        struct = secDisplay( data )
    return struct

class priDisplay:
    def __init__(self, data):
        self.ScanSig = []
        for dataIdx in data:
            self.ScanSig.append( scanAttr( dataIdx ) )
        self.tGraph = xyLable( 'Time (s)', 'Intensity (V)' )
        self.mzGraph = xyLable( 'm/z (Th)', 'Intensity (V)' )
    
class secDisplay:
    def __init__(self, data):
        self.IonSig = []
        for dataIdx in data:
            self.IonSig.append( ionAttr( dataIdx ) )
        self.ionGraph = xyLable( 'Time (s)', 'Amplitude' )

class xyLable:
    def __init__(self, xlableStr, ylabelStr ):
        self.xlabel = xlableStr
        self.ylabel = ylabelStr

class scanAttr:
    def __init__(self, data ):
        self.dt = data[0]
        self.xdata = data[1]
        self.ydata = data[2]
        
class ionAttr:
    def __init__(self, data ):
        self.dt = data[0]
        self.ydata = data[1]

