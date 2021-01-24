import os
import shutil
from xml.dom.minidom import parse
import cv2
import numpy as np
 
def readxml(path, name):
    domTree = parse(path)
    rootNood = domTree.documentElement
    objects = rootNood.getElementsByTagName('object')

    object_axis=[]

    for o in objects:
        object_name = o.getElementsByTagName('name')[0].childNodes[0].data
        bndbox = o.getElementsByTagName('bndbox')[0]
        xmin = int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data)
        ymin = int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data)
        xmax = int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data)
        ymax = int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data)

        if object_name == name:
            object_axis.append([name, xmin, ymin, xmax, ymax])
    
    return object_axis




root_path = r'/home/nick/Desktop/VHR10/'
Annotations = root_path + r'Annotations/'
JPEG = root_path + r'JPEGImages/'
classespath = root_path + r'classes1/'


classes = ['airplane','ship','storage tank','baseball diamond',
               'tennis court','basketball court','ground track field',
               'habor','bridge','vehicle']



files = os.listdir(Annotations)
files.sort(key=lambda x:int(x.split('.')[0]))
for fi in files:
    filename = fi.split('.')[0]
    imagename = filename+'.jpg'
    srcpath = JPEG+imagename
    

    for c in classes:
        savepath = classespath+c+'/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        axis = readxml(Annotations+fi, c)

        image = cv2.imread(srcpath)
        for i in axis:
            pt1 = (i[1],i[2])
            pt2 = (i[3],i[4])
            image = cv2.rectangle(image, pt1, pt2, (0,0,255), 2)
            if axis.index(i)==len(axis)-1:
                cv2.imwrite(savepath+imagename,image)
                print(fi)


        

