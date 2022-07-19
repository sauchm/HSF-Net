# HSF-Net
HSF-Net Image Restoration for Under-display Camera @MIPI-challenge ECCV,2022

Test:

Our test is divided into the following steps(note that path modification may occur in 1, 2, and 4)

1.python tool/n2p.py # makes NPY files convert to PNG format

2.python tool/32to24.py # makes the photo bit depth 24

3.python code/test.py # use the trained weight of 200 PTH, the results can be viewed in code/results/

4.python tool/p2n.py #makes the results in 3 into PNG format
