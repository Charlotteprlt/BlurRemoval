
from __future__ import division
import numpy 
import scipy
from numpy import *
from scipy import *
from pylab import *
import scipy.ndimage as ndimage

I = imread("car.png")[:,:,0] # The image is already in black &amp; white, so we can just read one color channel (here red)
# I is a N1 x N2 array such that I[i,j], between 0 and 1, codes the intensity of light at pixel (i,j).
# Careful: i is the vertical index and j is the horizontal one! (check it)

#figure()
#imshow(I, cmap="Greys_r") # Greys_r to display 0 as black and 1 and white

# useful functions: fft2 (not fft!), ifft2 (not ifft!), conj
figure()
imshow(log10(abs(fft2(I))), cmap="Greys_r") # Greys_r to display 0 as black and 1 and white
colorbar()

L = 15
h, w = I.shape

h_hyp = np.zeros((h,w))
for k in range(0, 2*L+1):
    h_hyp[0][k] = 1./(2*L)
print (h_hyp)

# test de h
I_flou_plus = ndimage.convolve(I, h_hyp)

figure()
imshow(I, cmap="Greys_r") 
show()
imshow(I_flou_plus, cmap="Greys_r")
show()

g_hat = fft2(I)
h_hat = fft2(h_hyp)
h_hat_conjugate = np.conjugate(h_hat)

mu = 0.01 * np.ones((h,w))
h_hat_abs_square = abs(h_hat)*abs(h_hat)

I_original_fou = (g_hat*h_hat_conjugate)/(h_hat_abs_square + mu)

f = ifft2(I_original_fou)

imshow(np.real(f), cmap="Greys_r")
show()