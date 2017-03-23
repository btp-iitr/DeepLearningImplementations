import sys
import tensorflow as tf
import skimage.transform
from skimage.color import rgb2gray, gray2rgb
from skimage.io import imsave, imread
from matplotlib import pyplot as plt
import numpy as np
import glob

filenames = sorted(glob.glob("pics/*.jpg"))

def load_image(path):
    img = imread(path)
    # crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
    # resize to 256, 256
    img = skimage.transform.resize(crop_img, (256, 256))
    return img
    # desaturate image
    #return (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3.0

def concat_images(imga, imgb):
    """
    Combines two image ndarrays side-by-side.
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.float32)
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return new_img

for i in range(len(filenames)):
    color_img = load_image(filenames[i])
    gray_img = rgb2gray(color_img)
    gray_img = gray2rgb(gray_img)
    su = concat_images(gray_img, color_img)
    plt.imsave("summary/" + '%04d'%i, su)
    print ("saved " + str(i))
