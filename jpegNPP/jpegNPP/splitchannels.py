import numpy as np
from skimage.color import rgb2ycbcr, ycbcr2rgb
from skimage.transform import resize
import sys
from skimage.util import img_as_float, img_as_ubyte
from skimage.io import imread


def main():
    filename = sys.argv[1]
    I = imread(filename)

    I.tofile('test.rgb.bin')
    
    I = img_as_float(I)

    

    print(I[0, 0], rgb2ycbcr(I[0, 0]*np.ones((1,1,1))))
    print(I[60, 60], rgb2ycbcr(I[60, 60]*np.ones((1,1,1))))
    print(I[60, 0], rgb2ycbcr(I[60, 0]*np.ones((1,1,1))))
    print(I[0, 60], rgb2ycbcr(I[0, 60]*np.ones((1,1,1))))
    print(I[120, 120], rgb2ycbcr(I[120, 120]*np.ones((1,1,1))))
    print(I[120, 0], rgb2ycbcr(I[120, 0]*np.ones((1,1,1))))
    
    Iy8 = rgb2ycbcr(I).astype(np.uint8)

    X = [0., 0., 0.] * np.ones((1, 1, 1))
    
    print('black, rgb', I[300, 300],
          'ycbcr', rgb2ycbcr(X),
          img_as_ubyte(ycbcr2rgb(rgb2ycbcr(X)))
    )
    
    Iy8[:,:,0].tofile(f'{filename}.y.bin')
    
    Icb = (resize(Iy8[:,:,1], (240, 320)) * 255).astype(np.uint8)
    Icr = (resize(Iy8[:,:,2], (240, 320)) * 255).astype(np.uint8)
    
    Icb.tofile(f'{filename}.cb.res.bin')
    Icr.tofile(f'{filename}.cr.res.bin')

    
if __name__ == '__main__':
    main()
