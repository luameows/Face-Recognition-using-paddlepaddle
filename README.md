# Face-Recognition-using-paddlepaddle
首届生物特征识别大赛，科赛网。
最终名次--第8，准确率94.4%。
## 预处理
1.采用mtcnn进行人脸裁剪并对齐，尽可能保留相关元素。

2.采用随机裁剪、色彩增强等方式对数据集进行上采样扩充，尽量控制数据集均衡。实验证明，均衡后的数据集训练效果更好，但是由于数据集更大，训练时间更长。

## 训练网络
采用sphereface20-net，效果比较理想。

学习率采用warmup策略，结合early stop，有不错提升。由于数据集较大，batch size=512需要跑1300多个batch才一轮epoch，最终未能训练到最优。

整个过程主要调整参数为lr, weight decay。

## 人脸验证
采用mirror trick测试，最后max out会有较好结果，大概能提升2%~



项目地址：[我的项目](https://www.kesci.com/home/project/5b713833a537e0001005beae)

详细博客地址： [我的博客]()
