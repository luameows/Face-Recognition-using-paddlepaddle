# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank
import cv2
from mtcnn.mtcnn import MTCNN
detector=MTCNN()
import os,sys
import math

rootpath1='/mnt/datasets/WebFace_fusai/train_set/'#训练集路径
rootpath2='/mnt/datasets/WebFace_fusai/validate_set/'#val集路径
rootpath3='/mnt/datasets/WebFace_fusai/testing_set/'
savepath='/home/kesci/dataset_fusai/train_val/'#预处理后保存图片路径

def tformfwd(trans, uv):
    uv = np.hstack((uv, np.ones((uv.shape[0], 1))))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy

def tforminv(trans, uv):
    Tinv = inv(trans)
    xy = tformfwd(Tinv, uv)
    return xy

def findNonreflectiveSimilarity(uv, xy, options=None):
    options = {'K': 2}
    K = options['K']
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))
    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')
    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]
    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ])
    T = inv(Tinv)
    T[:, 2] = np.array([0, 0, 1])
    return T, Tinv

def findSimilarity(uv, xy, options=None):
    options = {'K': 2}
    # Solve for trans1
    trans1, trans1_inv = findNonreflectiveSimilarity(uv, xy, options)
    # Solve for trans2
    # manually reflect the xy data across the Y-axis
    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]
    trans2r, trans2r_inv = findNonreflectiveSimilarity(uv, xyR, options)
    # manually reflect the tform to undo the reflection done on xyR
    TreflectY = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    trans2 = np.dot(trans2r, TreflectY)
    # Figure out if trans1 or trans2 is better
    xy1 = tformfwd(trans1, uv)
    norm1 = norm(xy1 - xy)
    xy2 = tformfwd(trans2, uv)
    norm2 = norm(xy2 - xy)
    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = inv(trans2)
        return trans2, trans2_inv

def get_similarity_transform(src_pts, dst_pts, reflective=True):
    if reflective:
        trans, trans_inv = findSimilarity(src_pts, dst_pts)
    else:
        trans, trans_inv = findNonreflectiveSimilarity(src_pts, dst_pts)
    return trans, trans_inv

def cvt_tform_mat_for_cv2(trans):
    cv2_trans = trans[:, 0:2].T
    return cv2_trans

def get_similarity_transform_for_cv2(src_pts, dst_pts, reflective=True):
    trans, trans_inv = get_similarity_transform(src_pts, dst_pts, reflective)
    cv2_trans = cvt_tform_mat_for_cv2(trans)
    return cv2_trans
'''判断是否是满足要求人脸
若置信度小于0.75，或图片太小---False
'''
def CheckFace(detect_info,index,img_w,img_h):
    box_x=detect_info[index]['box'][0]
    box_y=detect_info[index]['box'][1]
    width=detect_info[index]['box'][2]
    height=detect_info[index]['box'][3]
    confidence=detect_info[index]['confidence']
    if confidence<0.75:
        return False
    if width<60 or height<70:
        return False
    centerx=box_x+width/2
    centery=box_y+height/2
    if centerx<img_w/3 or centerx>img_w*2/3 or centery<img_h/3 or centery>img_h*2/3:
        return False
    return True

def GetFaceIndex(detect_info,img):
    detect_num=len(detect_info)
    face_index=-1
    img_w=img.shape[1]
    img_h=img.shape[0]
    if detect_num==0:
        return face_index
    if detect_num==1:
        if not CheckFace(detect_info,0,img_w,img_h):
            return face_index
        face_index=0
        return face_index
    if detect_num>1:
        index_all=[]
        for i in range(detect_num):
            if CheckFace(detect_info,i,img_w,img_h):
                index_all.append(i)
        if len(index_all)==0:
            return -1
        if len(index_all)==1:
            return index_all[0]
        dist=[]
        for i in index_all:
            box_x=detect_info[i]['box'][0]
            box_y=detect_info[i]['box'][1]
            weight=detect_info[i]['box'][2]
            height=detect_info[i]['box'][3]
            centerx=box_x+weight/2
            centery=box_y+height/2
            distance=math.pow(centerx-weight/2,2)+math.pow(centery-height/2,2)
            dist.append(distance)
        return index_all[dist.index(min(dist))]

def Alignment(img,facila5points):
    of=2
    ref_pts = [ [30.2946+of, 51.6963+of],[65.5318+of, 51.5014+of],
        [48.0252+of, 71.7366+of],[33.5493+of, 92.3655+of],[62.7299+of, 92.2041+of] ]
    crop_size = (96+of*2, 112+of*2)
    s=np.array(facial5points).astype(np.float32)
    r=np.array(ref_pts).astype(np.float32)
    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(img, tfm, crop_size)
    return face_img
    
for img_name in os.listdir(rootpath3):
    finalpath='/home/kesci/dataset_fusai/test/'
    if img_name=='.DS_Store':
        continue
    img=cv2.imread(rootpath3+img_name)
    detect_info=detector.detect_faces(img)
    face_index=GetFaceIndex(detect_info,img)
    if face_index==-1:
        continue
    else:
        face_info=detect_info[face_index]
        face_box=np.array(face_info['box'])
        face_leye=np.array(face_info['keypoints']['left_eye'])
        face_reye=np.array(face_info['keypoints']['right_eye'])
        face_nose=np.array(face_info['keypoints']['nose'])
        face_lmouth=np.array(face_info['keypoints']['mouth_left'])
        face_rmouth=np.array(face_info['keypoints']['mouth_right'])
        facial5points=np.array([face_leye,face_reye,face_nose,face_lmouth,face_rmouth])
        face_input=Alignment(img,facial5points)
        if not os.path.exists(finalpath):
            os.mkdir(finalpath)
        cv2.imwrite(finalpath+img_name,face_input)


for img_name in os.listdir('/home/kesci/dataset_fusai/test/'):
    imgpath='/home/kesci/dataset_fusai/test/'+img_name
    img=cv2.imread(imgpath)
    img = img[2:2+112,2:2+96,:]
    cv2.imwrite(imgpath,img)
        
print('finish test set, but some no-face imgs were dropped')

  
for class_num in os.listdir(rootpath1):
    path=rootpath1+class_num+'/'
    finalpath=savepath+class_num+'/'
    if class_num=='.DS_Store':
        continue
    for img_name in os.listdir(path):
        img=cv2.imread(path+img_name)
        detect_info=detector.detect_faces(img)
        face_index=GetFaceIndex(detect_info,img)
        if face_index==-1:
            continue
        else:
            face_info=detect_info[face_index]
            face_box=np.array(face_info['box'])
            face_leye=np.array(face_info['keypoints']['left_eye'])
            face_reye=np.array(face_info['keypoints']['right_eye'])
            face_nose=np.array(face_info['keypoints']['nose'])
            face_lmouth=np.array(face_info['keypoints']['mouth_left'])
            face_rmouth=np.array(face_info['keypoints']['mouth_right'])
            facial5points=np.array([face_leye,face_reye,face_nose,face_lmouth,face_rmouth])
            face_input=Alignment(img,facial5points)
            if not os.path.exists(finalpath):
                os.mkdir(finalpath)
            cv2.imwrite(finalpath+img_name,face_input)
print('finish train set')
            
for class_num in os.listdir(rootpath2):
    path=rootpath2+class_num+'/'
    finalpath=savepath+class_num+'/'
    if class_num=='.DS_Store':
        continue
    for img_name in os.listdir(path):
        img=cv2.imread(path+img_name)
        detect_info=detector.detect_faces(img)
        face_index=GetFaceIndex(detect_info,img)
        if face_index==-1:
            continue
        else:
            face_info=detect_info[face_index]
            face_box=np.array(face_info['box'])
            face_leye=np.array(face_info['keypoints']['left_eye'])
            face_reye=np.array(face_info['keypoints']['right_eye'])
            face_nose=np.array(face_info['keypoints']['nose'])
            face_lmouth=np.array(face_info['keypoints']['mouth_left'])
            face_rmouth=np.array(face_info['keypoints']['mouth_right'])
            facial5points=np.array([face_leye,face_reye,face_nose,face_lmouth,face_rmouth])
            face_input=Alignment(img,facial5points)
            if not os.path.exists(finalpath):
                os.mkdir(finalpath)
            cv2.imwrite(finalpath+img_name,face_input)
print('finish val set')

for img_name in os.listdir('/mnt/datasets/WebFace_fusai/testing_set/'):
    if img_name=='.DS_Store':
        continue
    imgpath='/home/kesci/dataset_fusai/test/'+img_name
    if not os.path.exists(imgpath):
        img=cv2.imread('/mnt/datasets/WebFace_fusai/testing_set/'+img_name)
        img = img[41:41+168,53:53+144,:]
        img=cv2.resize(img,(112,96))
        cv2.imwrite(imgpath,img)