import numpy as np
import math
import time
from numba import cuda, jit, njit, vectorize, int64, float64


#===================================================================#
@cuda.jit
def matMultiGPU(A, B, kernelLen, dataLen, C):
    x = cuda.grid(1)
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    if x >= dataLen[0]:
        return
    temp = 0
    for i in range( kernelLen[0] ):
        temp += B[ i + tx + bx*bw ] * A[i]
    C[x] = temp

    
#===================================================================#
# Convolution By CUDA
def Conv1D( kernel=[], ydata=[] ):
    kernel = np.array( kernel, dtype=np.float32 )
    ydata = np.array( ydata, dtype=np.float32 )

    if len(ydata) == 0:
        raise ValueError( 'data is empty' )
    
    if len(kernel) == 0:
        raise ValueError( 'kernel is empty' )
    
    # d_ --> device
    d_C = cuda.device_array( ydata.shape[0], np.float32 )
    
    fillData = np.zeros( int( np.floor( kernel.shape[0] / 2 ) ) )
    newData = np.hstack( ( fillData, ydata, fillData ) )
    d_A = cuda.to_device( kernel )  
    d_B = cuda.to_device( newData )
    d_kernelLen = cuda.to_device( kernel.shape[0] )
    d_dataLen = cuda.to_device( newData.shape[0] )
    
    threads = 32
    threadsPerBlock = (threads, 1)
    blocksPerGridX = math.ceil( newData.shape[0] / threads )
    blocksPerGrid = (blocksPerGridX, 1)
    
    matMultiGPU[ blocksPerGrid, threadsPerBlock ](d_A, d_B, d_kernelLen, d_dataLen, d_C)
    
    return d_C.copy_to_host()

# Convolution By CUDA
def PltConv1D( kernel , pltData ):

    pltName = type(pltData).__name__ 
    if pltName == 'priDisplay':
        for n, sig in enumerate( pltData.ScanSig ):
            pltData.ScanSig[n].ydata = Conv1D( kernel, sig.ydata )
    elif pltName == 'secDisplay':
        for n, sig in enumerate( pltData.IonSig ):
            pltData.IonSig[n].ydata = Conv1D( kernel, sig.ydata )
    else:
        raise TypeError("Source error")

    return pltData
