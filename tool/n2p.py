import os
import numpy as np
import matplotlib.image as img

def tonemap( x, alpha=0.25):
    mapped_x = x / (x + alpha)
    return mapped_x

imgs_path = 'C:\\Users\\86132\\Desktop\\4\\'
filename = os.listdir(imgs_path)
savepath ='C:\\Users\\86132\\Desktop\\3\\'

for i in range(len(filename)):
    img_name = filename[i].split('.')[0]+'.png'
    date = np.load(imgs_path + filename[i])
    date = tonemap(date)


    img.imsave(savepath + img_name,np.uint8(date*255.))

