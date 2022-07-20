# HSF-Net
HSF-Net Image Restoration for Under-display Camera @MIPI-challenge ECCVW,2022

# Prerequisites：

torch 1.7.0

python 3.8

opencv-python 4.2.0.32



# Test:

Our test is divided into the following steps(note that path modification may occur in step 1, 2, and 4)

1.python tool/n2p.py # makes NPY files convert to PNG format

2.python tool/32to24.py # makes the photo bit depth 24

3.python code/test.py # use the trained weight of 200.pth, the results can be viewed in code/results/

4.python tool/p2n.py #makes the results in 3 into NPY format


# Train: (note that 24 bit depth PNG format photos are required)


python code/train.py
