import cv2
import os,sys
import numpy as np
import random
fw=open('/home/kesci/work/augment.txt','w')
rootpath='/home/kesci/dataset_fusai/train_val/'
for dir_1 in os.listdir(rootpath):
    if dir_1=='.DS_Store':
        continue
    class_path=rootpath+dir_1+'/'
    nums=len(os.listdir(class_path))
    
    if nums<20:
        for img_name_jpg in os.listdir(class_path):
            imgpath=class_path+img_name_jpg
            imgname,_=img_name_jpg.split('.')
            img=cv2.imread(imgpath)
            img = np.array(img).astype('float32')
            img_flip = cv2.flip(img, 1)
            #随即裁剪2张,翻转1张
            rx = random.randint(0,2*2)
            ry = random.randint(0,2*2)
            
            img1 = img[ry:ry+112,rx:rx+96,:]
            rx = random.randint(0,2*2)
            ry = random.randint(0,2*2)
            img2 = img_flip[ry:ry+112,rx:rx+96,:]
            rx = random.randint(0,2*2)
            ry = random.randint(0,2*2)
            img3 = img_flip[ry:ry+112,rx:rx+96,:]
            #翻转图片与原始图片中心裁剪
            img_flip=img_flip[2:2+112,2:2+96,:]
            img = img[2:2+112,2:2+96,:]
            cv2.imwrite(class_path+imgname+'_1.jpg',img1)
            cv2.imwrite(class_path+imgname+'_2.jpg',img2)
            cv2.imwrite(class_path+imgname+'_4.jpg',img3)
            cv2.imwrite(class_path+imgname+'_3.jpg',img_flip)
            cv2.imwrite(imgpath,img)
            fw.write(class_path+imgname+'_1.jpg')
            fw.write('\n')
            fw.write(class_path+imgname+'_2.jpg')
            fw.write('\n')
            fw.write(class_path+imgname+'_3.jpg')
            fw.write('\n')
            fw.write(class_path+imgname+'_4.jpg')
            fw.write('\n')
    elif nums<40:
        for img_name_jpg in os.listdir(class_path):
            imgpath=class_path+img_name_jpg
            imgname,_=img_name_jpg.split('.')
            img=cv2.imread(imgpath)
            img = np.array(img).astype('float32')
            img_flip = cv2.flip(img, 1)
            #随即裁剪2张,翻转1张
            rx = random.randint(0,2*2)
            ry = random.randint(0,2*2)
            img1 = img[ry:ry+112,rx:rx+96,:]
            rx = random.randint(0,2*2)
            ry = random.randint(0,2*2)
            img2 = img_flip[ry:ry+112,rx:rx+96,:]
            #翻转图片与原始图片中心裁剪
            img_flip=img_flip[2:2+112,2:2+96,:]
            img = img[2:2+112,2:2+96,:]
            cv2.imwrite(class_path+imgname+'_1.jpg',img1)
            cv2.imwrite(class_path+imgname+'_2.jpg',img2)
            cv2.imwrite(class_path+imgname+'_3.jpg',img_flip)
            cv2.imwrite(imgpath,img)
            fw.write(class_path+imgname+'_1.jpg')
            fw.write('\n')
            fw.write(class_path+imgname+'_2.jpg')
            fw.write('\n')
            fw.write(class_path+imgname+'_3.jpg')
            fw.write('\n')
    elif nums<80:
        for img_name_jpg in os.listdir(class_path):
            imgpath=class_path+img_name_jpg
            imgname,_=img_name_jpg.split('.')
            img=cv2.imread(imgpath)
            img = np.array(img).astype('float32')
            img_flip = cv2.flip(img, 1)
            #翻转图片与原始图片中心裁剪
            img_flip=img_flip[2:2+112,2:2+96,:]
            img = img[2:2+112,2:2+96,:]
            cv2.imwrite(class_path+imgname+'_1.jpg',img_flip)
            cv2.imwrite(imgpath,img)
            fw.write(class_path+imgname+'_1.jpg')
            fw.write('\n')
    else:
        for img_name_jpg in os.listdir(class_path):
            imgpath=class_path+img_name_jpg
            imgname,_=img_name_jpg.split('.')
            img=cv2.imread(imgpath)
            img = np.array(img).astype('float32')
            img = img[2:2+112,2:2+96,:]
            cv2.imwrite(imgpath,img)
fw.close()