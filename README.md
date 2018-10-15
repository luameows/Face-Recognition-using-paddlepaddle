# Face-Recognition-using-paddlepaddle
首届生物特征识别大赛，科赛网。

最终名次，第8。准确率94.4%。
## 预处理
`pre/face_detect.py` \
1.采用mtcnn进行人脸裁剪并对齐，尽可能保留相关元素。

对于train，若检测多个人脸，选取距离图片中心最近，若未检测到人脸，丢弃。对于test，若未检测到人脸，强制中心裁剪。 \
处理后，train与val统一缩放为(96+4,112+4)，test强制中心裁剪(96*2,112*2)随后缩放为(96,112)。

`pre/calculate_mean_var.py` \
2.计算处理后图片均值与方差。

`pre/data_aug.py` \
3.采用随机裁剪、镜像翻转等方式对数据集进行上采样扩充，尽量控制数据集均衡。实验证明，均衡后的数据集训练效果更好，但是由于数据集更大，训练时间更长。

`pre/cleaned+list.txt` \
注： 该txt为清洗后的webface数据集索引。

## 训练网络
`train.py` \
采用sphereface20-net，效果比较理想。

学习率采用warmup策略，结合early stop，有不错提升。由于数据集较大，batch size=512需要跑1300多个batch才一轮epoch，最终未能训练到最优。

整个过程主要调整参数为lr, weight decay。

## 人脸验证
`infer.py` \
输出镜像前后feature，采用[mirror trick](https://github.com/happynear/NormFace/blob/master/MirrorFace.md)测试。

`acc.py` \
最后max out会有较好结果，大概能提升2%~



项目地址：[我的项目](https://www.kesci.com/home/project/5b713833a537e0001005beae)

详细博客地址： [我的博客](http://www.luameows.wang/2018/09/14/%E7%99%BE%E5%BA%A6%E7%94%9F%E7%89%A9%E7%89%B9%E5%BE%81%E8%AF%86%E5%88%AB%E5%A4%A7%E8%B5%9B-%E5%A4%8D%E8%B5%9B/#more)

# 
