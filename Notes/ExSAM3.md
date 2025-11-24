## 遥感图像分割测试
### 无人机视角
在无人机视角下SAM3的分割结果相较于遥感卫星图像的效果更好，但是也还存在优化空间

![UAVcar](https://raw.githubusercontent.com/YanghuiSong/SYH_GoGoGo/main/UploadImage/UAVcar.png)

![UAVbuildings](https://raw.githubusercontent.com/YanghuiSong/SYH_GoGoGo/main/UploadImage/UAVbuildings.png)

下图存在的问题是对于road（道路）这一概念理解的偏差，图中的分割结果显示停车场也被判断成为道路：

![UAVroad](https://raw.githubusercontent.com/YanghuiSong/SYH_GoGoGo/main/UploadImage/UAVroar.png)

SAM3几乎无法完成多类别分割，也就是全分割任务，又或者说无法识别复杂的提示词：

![UavCarRoad](https://raw.githubusercontent.com/YanghuiSong/SYH_GoGoGo/main/UploadImage/UAVroad%20and%20car.png)


### 卫星视角
下图存在的问题是分割不完全，在同样的低分辨率下农田的分割效果不佳，体现在相似的要素构成只能分割出少量的农田：

![field](https://raw.githubusercontent.com/YanghuiSong/SYH_GoGoGo/main/UploadImage/field.png)


![road](https://raw.githubusercontent.com/YanghuiSong/SYH_GoGoGo/main/UploadImage/road.png)

![roadandcar](https://raw.githubusercontent.com/YanghuiSong/SYH_GoGoGo/main/UploadImage/road%20and%20field.png)
