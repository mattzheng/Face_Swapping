# 平均脸
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
