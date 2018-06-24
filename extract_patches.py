#! /usr/bin/env python3

import os
import sys
import numpy as np
import glob
import uuid

from PIL import Image

from skimage.segmentation import slic
from skimage.measure import regionprops

for img in glob.glob(sys.argv[1] + '/**/*.jpg', recursive=True):
    print(img)
    im = Image.open(img)
    im = im.convert('L')
    im = im.resize((im.size[0]//8, im.size[1]//8))

    gt = Image.open(os.path.splitext(img)[0] + '.png')
    gt = gt.resize(im.size)
    # we don't care about border pixels
    gt = np.array(gt)[:,:,-1]

    sp = slic(im, n_segments=3000)
    # extract all centroids
    props = regionprops(sp)

    # extract 28x28 image patch from input and determine class for this
    # superpixel
    for prop in props:
        y = int(prop.centroid[0])
        x = int(prop.centroid[1])
        siz = 14
        patch = im.crop((x-siz, y-siz, x+siz, y+siz))

        # calculates highest non-1 
        def _max_value(mpatch):
            vals = [(3, 0b1000), (2, 0b0100), (1, 0b0010)]
            ret = 0
            m = 0
            for x in vals:
                if np.count_nonzero(np.bitwise_and(mpatch, x[1])) > m:
                    ret = x[0]
            return ret

        # calculates minimal class of centroid pixel
        def _min_value(c):
            vals = [(3, 0b1000), (2, 0b0100), (1, 0b0010)]
            ret = 0
            for x in vals:
                if np.bitwise_and(c, x[1]):
                    ret = x[0]
            return ret

        cls = _min_value(gt[y, x])
        #cls = _max_value(gt[sp == prop.label])
        patch.save('data/{}/{}.png'.format(cls, uuid.uuid4()))
