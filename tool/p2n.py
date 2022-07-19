import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

path = "C:\\Users\\86132\\Desktop\\4"
npy_list = os.listdir(path)
save_path = "C:\\Users\\86132\\Desktop\\3"
if not os.path.exists(save_path):
    os.mkdir(save_path)

for i in range(0, len(npy_list)):
    #print(i)
    #print(npy_list[i])
    npy_full_path = os.path.join(path, npy_list[i]) 
 
    save_full_path = os.path.join(save_path, npy_list[i][:-4])
    
    raw_image=Image.open(npy_full_path)
    image_array=np.array(raw_image, dtype='float32')/255

    if (1-image_array).all() != 0:

        image_array = (0.25 * image_array) / np.abs(1 - image_array)
        print("111")
    else:
        #image_array =np.maximum((1 - image_array), 0.3)
        image_array = (0.25 * image_array) / (np.abs(1 - image_array)+1e-39)
        print("aaa")

    #print(image_array.dtype)
    np.save(save_full_path,image_array)