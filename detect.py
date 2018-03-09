# -*- coding: utf-8 -*-

import cv2
import sys
import os.path
from glob import glob

def detect(filename, cascade_file="/home/veetsin/anaconda3/envs/gluon/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    if image is None: # read failure and skip this image
      return
    if image.shape[2]==1: # drop gray images
      return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=3,
                                     minSize=(96, 96))
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y: y + h, x:x + w, :]
        face = cv2.resize(face, (96, 96))
        save_filename = '%s-%d.jpg' % (os.path.basename(filename).split('.')[0], i)
        cv2.imwrite("face/" + save_filename, face)


if __name__ == '__main__':
    folder_name = os.listdir()
    if os.path.exists('face') is False:
        os.makedirs('face')
    for name in folder_name:
        file_list = glob('%s/*.jp*'%(name))
        for filename in file_list:
            print(filename)
            detect(filename)