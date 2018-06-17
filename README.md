Patchwise pixel labelling for segmentation
------------------------------------------

> I do not consider this method suitable for a general purpose segmenter as the
> superpixel segmentation is not suitable for non-Latin scripts and semantic
> segmentation still requires the whole traditional line extraction pipeline.

A quick and dirty reimplementation of the shallow CNN superpixel labelling
method for semantic segmentation presented in [0].

`extract_patches.py` takes a directory with jpg source images and png pixel
labelled images and extracts 28x28 image patches oriented around the centroids
of a SLIC oversegmentation. The class of each patch is derived from the highest
assigned class of the centroid pixel (alternatively there is a frequency
selection method implemented). Output is written to data/{0,1,2,3}/$uuid.png

`main.py` is used for training. Training is quite fast (~30s/100k samples) on a
modern machine.

Dataset
-------

Download the DIVA HisDB from [1] image files and pixel-labelled image files.

Additional tools needed
-----------------------

Pixel labellings/semantic segmentation (really a misnomer) are not immediately
useful for line extraction for recognition, as the line separation/instance
detection problem is not solved by these systems. The real world use of this
approach requires a good binarization to project the predicted superpixel
classes onto foreground pixels and an actual extractor such as a seam carver or
the less-than-stellar one in kraken. The main reduction in complexity comes
from being able to separate interlinear components, effectively increasing line
spacing in challenging manuscripts.

Differences
-----------

- momentum in SGD.

[0] Chen, Kai, et al. "Convolutional neural networks for page segmentation of historical document images." Document Analysis and Recognition (ICDAR), 2017 14th IAPR International Conference on. Vol. 1. IEEE, 2017.
[1] http://diuf.unifr.ch/main/hisdoc/diva-hisdb
