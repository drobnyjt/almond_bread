import numpy as np
from PIL import Image,ImageDraw,ImageOps

max_iter = 100

def f(c, n):
    x = 0
    for i in range(max_iter):
        x = x**1.75 + c
        if (abs(x) > 2.0):
            break
    return i

width = 600*4
height = 600*4
x0 = 1.0
y0 = 0.5j
zoom = 0.6
dx = 1./zoom
dy = 1./zoom
sx = dx/width
sy = dy/height
num = 5

#width = 600
#height = 600
#x0 = 0.9
#y0 = 0.3j
#zoom = 6.
#dx = 1./zoom
#dy = 1./zoom
#sx = dx/width
#sy = dy/height
#num = 5

color = np.zeros(shape=(width,height),dtype='complex64')

for i in range(width):
    for k in range(height):
        c = sx*i - x0 + sy*k*1j - y0
        color[i,k] = f(c,5)

color /= np.max(color)
color = np.conj(color)*color
#im = Image.new('RGB', (width, height), (0, 0, 0))

#im = Image.fromarray(np.array([color.astype('float32')*255, np.ones((width,height)), np.ones((width,height))]),'HSV')
im = Image.fromarray(color.astype('float32')*255).convert('RGB')
im = im.convert('HSV')
h,s,v = im.split()
array = 255*np.ones((width,height))
h = Image.fromarray(array).convert('RGB')
h,s,test = h.split()

im = Image.merge('HSV', (v,h,ImageOps.invert(v)))
im.show()
