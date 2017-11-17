# Face_Swapping

简单换脸、人脸对齐、关键点定位与画图

这是一个利用dlib进行关键点定位 + opencv处理的人脸对齐、换脸、关键点识别的小demo。原文来自于[《Switching Eds: Face swapping with Python, dlib, and OpenCV》](https://matthewearl.github.io/2015/07/28/switching-eds-with-python/)
该博文的[github](https://github.com/matthewearl/faceswap/blob/master/faceswap.py)地址中有所有的code。这边我将我抽取的code放在自己的github之中，可以来这下载:
https://github.com/mattzheng/Face_Swapping

有人将其进行[中文翻译](http://python.jobbole.com/82546/)也有将其进行一定改编有以下两个案例：

 - 1.[《川普撞脸希拉里(基于 OpenCV 的面部特征交换)-2》](http://blog.csdn.net/oxuzhenyi/article/details/54982632)
 - [变脸](http://messcode.github.io/2016/04/17/switch-faces-using-python/)

### 变脸贴图：
从这张：
![这里写图片描述](http://7xrpb1.com1.z0.glb.clouddn.com/marked_img.jpg)
变为这张：
![这里写图片描述](http://7xrpb1.com1.z0.glb.clouddn.com/switched_face.jpg)

因为原文里面内容丰富，我觉得可以提取出很多有用的小模块，于是乎：
.

提取一：关键点定位与画图
============

```
import cv2
import dlib
import numpy
import sys
import matplotlib.pyplot as plt
SCALE_FACTOR = 1 # 图像的放缩比

def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s

def annotate_landmarks(im, landmarks):
    '''
    人脸关键点，画图函数
    '''
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im
```

然后实践就是载入原图：

```
im1, landmarks1 = read_im_and_landmarks('02.jpg')  # 底图
im1 = annotate_landmarks(im1, landmarks1)

%matplotlib inline
plt.subplot(111)
plt.imshow(im1)
```
.

提取二：人脸对齐
========

需要一张模板图来作为靠拢的对象图。

```
# 人脸对齐函数
def face_Align(Base_path,cover_path):
    im1, landmarks1 = read_im_and_landmarks(Base_path)  # 底图
    im2, landmarks2 = read_im_and_landmarks(cover_path)  # 贴上来的图
    
    if len(landmarks1) == 0 & len(landmarks2) == 0 :
        raise ImproperNumber("Faces detected is no face!")
    if len(landmarks1) > 1 & len(landmarks2) > 1 :
        raise ImproperNumber("Faces detected is more than 1!")
    
    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])
    warped_im2 = warp_im(im2, M, im1.shape)
    return warped_im2
```
#### 这里的步骤是：

 - 提取模板图、对齐图的landmarks; 
 - 通过transformation_from_points计算对齐图向模板图的转移矩阵M，变换矩阵是根据以下公式计算出来的;
 - warp_im，将 im2 的掩码进行变化，使之与 im1 相符

实践的话就是：

```
FEATHER_AMOUNT = 19  # 匹配的时候，特征数量，现在是以11个点为基准点  11  15  17 

Base_path = '01.jpg'
cover_path = '02.jpg'
warped_mask = face_Align(Base_path,cover_path)
```
.

提取三：换脸
======

主要函数：
```
def Switch_face(Base_path,cover_path):
    im1, landmarks1 = read_im_and_landmarks(Base_path)  # 底图
    im2, landmarks2 = read_im_and_landmarks(cover_path)  # 贴上来的图
    
    if len(landmarks1) == 0 & len(landmarks2) == 0 :
        raise ImproperNumber("Faces detected is no face!")
    if len(landmarks1) > 1 & len(landmarks2) > 1 :
        raise ImproperNumber("Faces detected is more than 1!")
    
    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])
    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],
                              axis=0)
    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    return output_im
```
#### 主要步骤：
 - 提取模板图、对齐图的landmarks; 
 - M，通过transformation_from_points计算对齐图向模板图的转移矩阵M;
 

```
matrix([[   0.62876962,    0.20978991, -101.32973923],
        [  -0.20978991,    0.62876962,   79.11235991],
        [   0.        ,    0.        ,    1.        ]])
```

 - mask,得到基于对齐图的掩膜，get_face_mask函数，获取 im2 的面部掩码，mask长成这样：
 ![这里写图片描述](https://matthewearl.github.io/assets/switching-eds/mask.png)
 - warped_mask ,warp_im函数，将 im2 的掩码进行变化，使之与 im1 相符,跟上面的mask张一样（一个鼻子）
 - combined_mask ，将二者的掩码进行连通（跟warped_mask 长一样）
 - warped_im2 ，warp_im函数，第二次，将第二幅图像调整到与第一幅图像相符（对齐图片,斜了点）
![这里写图片描述](http://img.blog.csdn.net/20171117184429306?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMjY5MTczODM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
 - warped_corrected_im2 ，correct_colours函数，将 im2 的皮肤颜色进行修正，使其和 im1 的颜色尽量协调（类似下图）
 ![这里写图片描述](http://img.blog.csdn.net/20171117184614032?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMjY5MTczODM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
 - output_im  组合图像，获得结果

实践：

```
FEATHER_AMOUNT = 23

Base_path = '03.jpg'
cover_path = '02.jpg'
output_im = Switch_face(Base_path,cover_path)
```
