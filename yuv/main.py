import numpy as np
from skimage.color import rgb2yuv, yuv2rgb
import matplotlib.pyplot as plt


def fun(Iout, I, idx, n):        
    p = idx

    print(p)
    
    rows, cols = I.shape[:-1]

    while (p < (rows * cols)):
            
        i, j = p//cols, p % cols
        
        Iout[i, j, 0] = .299 * I[i, j, 0] + .587 * I[i, j, 1] + 0.114 * I[i, j, 2]
        Iout[i, j, 1] = -.14713 * I[i, j, 0] -.28886 * I[i, j, 1] + .436 * I[i, j, 2]
        Iout[i, j, 2] = .615 * I[i, j, 0] -.51499 * I[i, j, 1] -.10001 * I[i, j, 2]

        p += n


import datetime
            
def loop(I):    
    # Simple loop
    rows, cols = I.shape[:-1]
    
    Iout = np.zeros(I.shape)

    start = datetime.datetime.now()
    
    for i in range(rows):
        for j in range(cols):
            Iout[i, j, 0] = .299 * I[i, j, 0] + .587 * I[i, j, 1] + 0.114 * I[i, j, 2]
            Iout[i, j, 1] = -.14713 * I[i, j, 0] -.28886 * I[i, j, 1] + .436 * I[i, j, 2]
            Iout[i, j, 2] = .615 * I[i, j, 0] -.51499 * I[i, j, 1] -.10001 * I[i, j, 2]

    end = datetime.datetime.now()
    
    print(Iout[0, 0, :], (end - start).total_seconds() * 1e3, 'ms')

    # Threads
    Iout2 = np.zeros(I.shape)
    
    from threading import Thread

    n_threads = 2
    threads = [Thread(target=fun, args=(Iout2, I, idx, n_threads))
               for idx in range(n_threads)]

    start = datetime.datetime.now()
    
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    end = datetime.datetime.now()
        
    print('Iout2', Iout2[0, 0, :], (end - start).total_seconds() * 1e3, 'ms')

    
def main():
    I = np.array([
	[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
	[[255, 255, 255], [0, 0, 0], [128, 128, 128]]
    ], dtype=np.uint8)

    plt.figure('I')
    plt.imshow(I)

    plt.figure('Iyuv')
    plt.imshow(np.clip(rgb2yuv(I), 0, 1))

    I1 = I/255.    
    Iout = np.zeros_like(I1)

    print(Iout.dtype, I1.dtype)
    
    for i in range(I1.shape[0]):
        for j in range(I1.shape[1]):
            Iout[i, j, 0] = .299 * I1[i, j, 0] + .587 * I1[i, j, 1] + 0.114 * I1[i, j, 2]
            Iout[i, j, 1] = -.14713 * I1[i, j, 0] -.28886 * I1[i, j, 1] + .436 * I1[i, j, 2]
            Iout[i, j, 2] = .615 * I1[i, j, 0] -.51499 * I1[i, j, 1] -.10001 * I1[i, j, 2]

    print(Iout)

    exit()

    plt.figure('Iyuv custom')
    plt.imshow(np.uint8(np.clip(Iout, 0, 1) * 255))

    Iout1 = np.zeros_like(I)
    
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):

            R = I[i, j, 0]
            G = I[i, j, 1]
            B = I[i, j, 2]
            
            Y = ( (  66 * R + 129 * G +  25 * B + 128) >> 8) +  16
            U = ( ( -38 * R -  74 * G + 112 * B + 128) >> 8) + 128
            V = ( ( 112 * R -  94 * G -  18 * B + 128) >> 8) + 128

            Iout1[i, j] = np.array([Y, U, V])
            
            # Iout1[i, j, 0] = ((66*I[i, j, 0]+129*I[i, j, 1]+ 25*I[i, j, 2] + 128) >> 8) + 16
            # Iout1[i, j, 1] = ((-38*I[i, j, 0]-74*I[i, j, 1]+112*I[i, j, 2] + 128) >> 8) + 128
            # Iout1[i, j, 2] = ((112*I[i, j, 0]-94*I[i, j, 1]-18*I[i, j, 2] + 128) >> 8) + 128

            
    Irgb = yuv2rgb(Iout)
            
    plt.figure('Irgb from yuv')
    plt.imshow(np.clip(Irgb, 0, 1))
    
    plt.show()

    
if __name__ == '__main__':
    main()
