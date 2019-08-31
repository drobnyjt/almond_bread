import numpy as np
from PIL import Image,ImageDraw,ImageOps
from numba import jit, float64, int32, prange
import time
import imageio

@jit('int32(complex128, float64, float64, int32)', nopython=True, nogil=True)
def mandelbrot_generator(c, power, threshold, max_iter):
    x = 0
    for i in range(max_iter):
        x = x**power + c
        if (np.abs(x) > threshold):
            break
    return i

@jit('float64[:,:](float64, float64, float64, float64, int32, int32, float64, float64, int32)',
    nopython=False, parallel=False, nogil=False)
def render_set(width, height, x0, y0, image_width, image_height, power, threshold, max_iter):

    color = np.zeros((image_width, image_height), dtype=np.float64)
    dx = width / image_width
    dy = height / image_height

    #for i, k in itertools.product(range(width), range(height)):
    for i in prange(image_width):
        for k in prange(image_height):
            c = (x0 - width/2. + dx*i) + (y0 - height/2. + dy*k)*1j

            color[i, k] = mandelbrot_generator(c, power, threshold, max_iter)

    return color

def normalize_color_array_in_place(color):
    color[:,:] -= np.min(color)
    color[:,:] /= np.max(color)

def process_and_save_image(color, image_index, reduction_factor=1):
    width, height = np.shape(color)
    #Generate image from array, and split into HSV channels
    im = Image.fromarray(color*255).convert('RGB').convert('HSV')
    h,s,v = im.split()

    #Generate white image and split into HSV channels
    full_array = 255*np.ones((width,height))
    white_image = Image.fromarray(full_array).convert('RGB')
    white_h, white_s, white_v = white_image.split()

    #Combine Mandelbrot V channel and blank channel into final image
    im = Image.merge('HSV', (v, white_h, ImageOps.invert(v))).convert('RGB')

    #Reduce, save, and return image using downsampling algorithm
    im = ImageOps.fit(im, (width//reduction_factor, height//reduction_factor), method=Image.LANCZOS)
    im.save('mandelbrot_'+str(image_index)+'.png', 'png')
    return im

def main():
    width = 0.0001
    height = 0.0001
    image_width = 2000
    image_height = 2000

    x0 = -0.761574
    y0 = -0.0847596

    power = 2.
    threshold = 2.
    max_iter = 200

    sizes = np.logspace(1, -4, 1000)

    for index, size in enumerate(sizes):
        print('Frame ', index)
        width = size
        height = size
        color = render_set(width, height, x0, y0, image_width, image_height, power, threshold, max_iter)
        normalize_color_array_in_place(color)
        image = process_and_save_image(color, index)
        #image.show()


    images = []
    for index, _ in enumerate(sizes):
        file = 'mandelbrot_'+str(index)+'.png'
        images.append(imageio.imread(file))

    imageio.mimsave('mandelbrot_movie.gif', images, duration=1/30.)

if __name__ == '__main__':
    main()
