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





Annotations = r'/home/nick/Desktop/DIOR/Annotations/'
JPEG = r'/home/nick/Desktop/DIOR/JPEGImages/'
classespath = r'/home/nick/Desktop/DIOR/classes1/'


classes = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
'chimney', 'dam', 'Expressway-Service-area', 'golffield', 'groundtrackfield',
'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 
'trainstation', 'vehicle', 'windmill', 'Expressway-toll-station']

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


        


    # f = open(Annotations+fi)
    # lines = f.readlines()
    # for line in lines:
    #     for c in classes:
    #         if c in line:
    #             savepath = classespath+c+'/'
    #             if not os.path.exists(savepath):
    #                 os.makedirs(savepath)
    #             srcpath = JPEG+imagename
    #             if not os.path.exists(srcpath):
    #                 continue
    #             shutil.copyfile(srcpath,savepath + imagename)