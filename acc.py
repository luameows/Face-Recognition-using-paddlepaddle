# -*- coding: utf-8 -*-
import os,sys
import numpy as np

features=np.loadtxt('/home/kesci/features.txt')
features_mirror=np.loadtxt('/home/kesci/features_mirror.txt')

add_features=features+features_mirror
max_features=np.maximum(features,features_mirror)

def cosDistance(array_encodings):
    dist_cos=[]
    length=int (len(array_encodings)/2)
    for i in range(0, length):
        num=np.dot(array_encodings[2*i],array_encodings[2*i+1])
        cos=num/(np.linalg.norm(array_encodings[2*i])*np.linalg.norm(array_encodings[2*i+1]))
        sim=0.5+0.5*cos 
        dist_cos.append(sim)
    return dist_cos


dist_cost=cosDistance(max_features)
with open('/mnt/datasets/WebFace_fusai/pair_id_for_users.txt') as flist:
    lines = [line.strip() for line in flist]
    lines=lines[1:]
    csvFile = open('/home/kesci/result.csv','wb') 
    writer = csv.writer(csvFile)
    writer.writerow(['submit_pairsID','prob'])
    for i in range(len(dist_cost)):
        writer.writerow([lines[i],dist_cost[i]])
    csvFile.close()