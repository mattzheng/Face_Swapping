# 项目一：Face_Swapping

简单换脸、人脸对齐、关键点定位与画图

这是一个利用dlib进行关键点定位 + opencv处理的人脸对齐、换脸、关键点识别的小demo。原文来自于[《Switching Eds: Face swapping with Python, dlib, and OpenCV》](https://matthewearl.github.io/2015/07/28/switching-eds-with-python/)
该博文的[github](https://github.com/matthewearl/faceswap/blob/master/faceswap.py)地址中有所有的code。这边我的博客地址：
http://blog.csdn.net/sinat_26917383/article/details/78564416

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
 - warped_corrected_im2 ，correct_colours函数，将 im2 的皮肤颜色进行修正，使其和 im1 的颜色尽量协调（类似下图）

 - output_im  组合图像，获得结果

实践：

```
FEATHER_AMOUNT = 23

Base_path = '03.jpg'
cover_path = '02.jpg'
output_im = Switch_face(Base_path,cover_path)
```


# 项目二：平均脸
新更新了一个平均脸的程序内容：
![这里写图片描述](https://www.learnopencv.com/wp-content/uploads/2016/05/average_best_actress-300x300.jpg)

py代码以及相关数据地址：https://www.learnopencv.com/wp-content/uploads/2016/05/FaceAverage.zip
最初博文地址：https://www.learnopencv.com/average-face-opencv-c-python-tutorial/ 
中文翻译：http://blog.csdn.net/GraceDD/article/details/51382952
中文改编地址：[《手把手：用OpenCV亲手给小扎、Musk等科技大佬们做一张“平均脸”（附Python代码）》](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651654758&idx=1&sn=b60e2da0b4e9cffed660f44bd624eb9e&chksm=bd4c2df58a3ba4e32f938df33cdc780bd7041087c6f3b82cf059c036a50c12e97067a8d12815&mpshare=1&scene=1&srcid=1123eFDjNTtDFdq4GS8M2e8d#rd)

在完成各个library的安装后。 

 - 第一步：将要平均的照片放入faces文档，确保图片为jpg格式。
 - 第二步：在终端运行 python face_landmark_detection.py
   shape_predictor_68_face_landmarks.dat
   faces，并在程序运行结束后将所有faces文档中的文件复制到presidents文档中（如无法完成dlib安装，可略过该步骤，直接用文摘菌提供的素材）
 - 第三步：在终端运行 python faceAverage.py 这样就能看到制作成功的平均脸了！

具体实现步骤:

 - 1.读入图 + 读入关键点信息  readPoints  readImages
 - 2.平均脸的眼角位置（这样其他脸，按照眼睛位置对齐）  eyecornerDst
 - 3.新的8个初始边界点 boundaryPts （为了后续做脸谱网络用的）
 - 4.设置初始平均脸 pointsAvg （随便找个脸68个关键点 + 8个初始点）
 - 5.根据眼睛位置，进行人脸初步对齐
 - 6.计算初始平均脸的脸谱网络76点（calculateDelaunayTriangles）
 - 7.根据脸谱网络二次人脸对齐
![这里写图片描述](https://www.learnopencv.com/wp-content/uploads/2016/05/image-warping-based-on-delaunay-triangulation-768x256.jpg)
本案例中进行了两次对齐，眼睛对齐之后，通过Warp Triangles 再此对齐。
