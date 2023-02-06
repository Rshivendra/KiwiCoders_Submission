---
title: 基于OpenCv的人脸识别
date: 2021/6/21 8:22:00
tags: 
  - Python
  - 深度学习 | deep learning
  - OpenCv
categories: 人工智能 | artificial intelligence
description: Opencv是一个开源的的跨平台计算机视觉库，内部实现了图像处理和计算机视觉方面的很多通用算法，对于python而言，在引用opencv库的时候需要写为import cv2。其中，cv2是opencv的C++命名空间名称，使用它来表示调用的是C++开发的opencv的接口 | 
Opencv is an open source cross-platform computer vision library, which internally implements many general algorithms for image processing and computer vision. For python, it needs to be written as import cv2 when referencing the opencv library. Among them, cv2 is the C++ namespace name of opencv, which is used to indicate that the opencv interface developed by C++ is called
---





> 实验环境 | lab enviroment：python 3.6 + opencv-python 3.4.14.51
> 建议使用 | recommended to use anaconda configure to the same enviroment

# 背景 | background

## 人脸识别步骤 | face recognition steps

<img src="https://gitee.com/Cheney822/images/raw/master/PicGo/20210622184054832.png" alt="image-20220311161053996" style="zoom: 50%;"/>





###  人脸采集 | face collection
采集人脸图片的方法多种多样，可以直接从网上下载数据集，可以从视频中提取图片，还可以从摄像头实时的采集图片。 | 
there are many ways to collect face pictures. You can download data sets directly from the Internet, extract pictures from videos, and collect pictures in real time from cameras.

###  人脸检测方法 | face detection method
人脸检测在实际中主要用于人脸识别的预处理，即在图像中准确标定出人脸的位置和大小。人脸图像中包含的模式特征十分丰富，如直方图特征、颜色特征、模板特征、结构特征及Haar特征等。人脸检测就是把这其中有用的信息挑出来，并利用这些特征实现人脸检测。|
face detection is mainly used for preprocessing of face recognition in practice, that is, to accurately mark the position and size of the face in the image. The pattern features contained in the face image are very rich, such as histogram features, color features, template features, structural features, and Haar features. Face detection is to pick out the useful information and use these features to realize face detection.

###   人脸图像预处理 | face image processing
对于人脸的图像预处理是基于人脸检测结果，对图像进行处理并最终服务于特征提取的过程。系统获取的原始图像由于受到各种条件的限制和随机 干扰，往往不能直接使用，必须在图像处理的早期阶段对它进行灰度校正、噪声过滤等图像预处理。对于人脸图像而言，其预处理过程主要包括人脸图像的光线补 偿、灰度变换、直方图均衡化、归一化、几何校正、滤波以及锐化等。 |
image preprocessing for faces is the process of processing images based on face detection results and finally serving for feature extraction. The original image acquired by the system cannot be used directly due to various conditions and random interference. It must be pre-processed in the early stage of image processing, such as grayscale correction and noise filtering. For face images, the preprocessing process mainly includes light compensation, grayscale transformation, histogram equalization, normalization, geometric correction, filtering and sharpening of face images.

###  人脸特征提取 | face feature extraction
人脸识别系统可使用的特征通常分为视觉特征、像素统计特征、人脸图像变换系数特征、人脸图像代数 特征等。人脸特征提取就是针对人脸的某些特征进行的。人脸特征提取，也称人脸表征，它是对人脸进行特征建模的过程。人脸特征提取的方法归纳起来分为两大 类：一种是基于知识的表征方法；另外一种是基于代数特征或统计学习的表征方法。 |
the features that can be used by the face recognition system are usually divided into visual features, pixel statistical features, face image transformation coefficient features, face image algebraic features, etc. Face feature extraction is carried out for certain features of the face. Face feature extraction, also known as face representation, is the process of modeling the features of a face. The methods of face feature extraction can be summarized into two categories: one is representation method based on knowledge; the other is representation method based on algebraic features or statistical learning.


### 匹配与识别 | matching and identification
提取的人脸图像的特征数据与数据库中存储的特征模板进行搜索匹配，通过设定一个阈值，当相似度超过这一阈值，则把匹配得到的结果输 出。人脸识别就是将待识别的人脸特征与已得到的人脸特征模板进行比较，根据相似程度对人脸的身份信息进行判断。这一过程又分为两类：一类是确认，是一对一 进行图像比较的过程，另一类是辨认，是一对多进行图像匹配对比的过程。 |
the feature data of the extracted face image is searched and matched with the feature template stored in the database. By setting a threshold, when the similarity exceeds this threshold, the matching result is output. Face recognition is to compare the face features to be recognized with the obtained face feature templates, and judge the identity information of the faces according to the degree of similarity. This process is further divided into two categories: one is confirmation, which is a one-to-one image comparison process, and the other is identification, which is a one-to-many image matching and comparison process.

##  关于OpenCv | about OpenCv
Opencv是一个开源的的跨平台计算机视觉库，内部实现了图像处理和计算机视觉方面的很多通用算法，对于python而言，在引用opencv库的时候需要写为import cv2。其中，cv2是opencv的C++命名空间名称，使用它来表示调用的是C++开发的opencv的接口 |
Opencv is an open source cross-platform computer vision library, which internally implements many common algorithms in image processing and computer vision. For python, it needs to be written as import cv2 when referencing the opencv library. Among them, cv2 is the C++ namespace name of opencv, which is used to indicate that the opencv interface developed by C++ is called

目前人脸识别有很多较为成熟的方法，这里调用OpenCv库，而OpenCV又提供了三种人脸识别方法，分别是LBPH方法、EigenFishfaces方法、Fisherfaces方法。本文采用的是LBPH（Local Binary Patterns Histogram，局部二值模式直方图）方法。在OpenCV中，可以用函数cv2.face.LBPHFaceRecognizer_create()生成LBPH识别器实例模型，然后应用cv2.face_FaceRecognizer.train()函数完成训练，最后用cv2.face_FaceRecognizer.predict()函数完成人脸识别。 |
at present, there are many mature methods for face recognition. The OpenCv library is called here, and OpenCV provides three face recognition methods, namely the LBPH method, the EigenFishfaces method, and the Fisherfaces method. This article uses the LBPH (Local Binary Patterns Histogram, local binary pattern histogram) method. In OpenCV, you can use the function cv2.face.LBPHFaceRecognizer_create() to generate the LBPH recognizer instance model, then apply the cv2.face_FaceRecognizer.train() function to complete the training, and finally use the cv2.face_FaceRecognizer.predict() function to complete face recognition.

CascadeClassifier，是Opencv中做人脸检测的时候的一个级联分类器。并且既可以使用Haar，也可以使用LBP特征。其中Haar特征是一种反映图像的灰度变化的，像素分模块求差值的一种特征。它分为三类：边缘特征、线性特征、中心特征和对角线特征。|
CascadeClassifier is a cascade classifier for face detection in Opencv. And both Haar and LBP features can be used. Among them, the Haar feature is a feature that reflects the grayscale change of the image and calculates the difference value of the pixel module. It is divided into three categories: edge features, linear features, center features, and diagonal features.

# 程序设计 | programming

## 人脸识别算法 | face recognition algorithm：

<img src="https://gitee.com/Cheney822/images/raw/master/PicGo/20210622184156714.png"/>





### 1. 准备工作 | preparation

<img src="https://gitee.com/Cheney822/images/raw/master/PicGo/20210622184233123.png"/>





首先读取config文件，文件中第一行代表当前已经储存的人名个数，接下来每一行是二元组（id，name）即标签和对应的人名
读取结果存到以下两个全局变量中。 | 
first read the config file, the first line in the file represents the number of names currently stored, and each next line is a two-tuple (id, name) that is the label and the corresponding name
The read results are stored in the following two global variables.
```python
id_dict = {}  # 字典里存的是id——name键值对 | the dictionary stores id-name key-value pairs
Total_face_num = 999  # 已经被识别有用户名的人脸个数, | the number of faces with usernames that have been recognized,
```

def init():  # 将config文件内的信息读入到字典中 | read the information in the config file into the dictionary

加载人脸检测分类器Haar，并准备好识别方法LBPH方法 | load the face detection classifier Haar, and prepare the recognition method LBPH method

```python
# 加载OpenCV人脸检测分类器Haar | load OpenCV face detection classifier Haar | load OpenCV face detection classifier Haar | Load OpenCV face detection classifier Haar
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# 准备好识别方法LBPH方法 | ready to identify method LBPH method
recognizer = cv2.face.LBPHFaceRecognizer_create()
```
然后打开标号为0的摄像头 | then open the camera labeled 0

```python
camera = cv2.VideoCapture(0)  # 摄像头 | camera
success, img = camera.read()  # 从摄像头读取照片 | read photos from camera
```



### 2.录入新面容 | enter a new face



<img src="https://gitee.com/Cheney822/images/raw/master/PicGo/20210622184310674.png"/>







#### 2.1采集面容 | face collection
创建文件夹data用于储存本次从摄像头采集到的照片，每次调用前先清空这个目录。| 
create a folder data to store the photos collected from the camera this time, and clear this directory before each call.

然后是一个循环，循环次数为需要采集的样本数，摄像头拍摄取样的数量,越多效果越好，但获取以及训练的越慢。 | 
then there is a loop, the number of loops is the number of samples that need to be collected, the number of samples taken by the camera, the more the better, but the slower the acquisition and training

循环内调用`camera.read()`返回值赋给全局变量success,和img 用于在GUI中实时显示。 | 
the return value of `camera.read()` called in the loop is assigned to the global variable success, and img is used for real-time display in the GUI.

然后调用`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`用于将采集到的图片转为灰度图片减少计算量。 | 
then call `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` to convert the collected image into a grayscale image to reduce the amount of calculation.

然后利用加载好的人脸分类器将每一帧摄像头记录的数据带入OpenCv中，让Classifier判断人脸。| 
then use the loaded face classifier to bring the data recorded by each frame of the camera into OpenCv, and let the Classifier judge the face.

```python
  # 其中gray为要检测的灰度图像，1.3为每次图像尺寸减小的比例，5为minNeighbors | where gray is the grayscale image to be detected, 1.3 is the ratio of each image size reduction, and 5 is minNeighbors
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
```
faces为在img图像中检测到的人脸，然后利用cv2.rectangle在人脸一圈画个矩形。并把含有人脸的区域储存进入data文件夹 | 
faces is the face detected in the img image, and then use cv2.rectangle to draw a rectangle around the face. And store the area containing the face into the data folder

注意这里写入时，每个图片的标签时`Total_face_num`即当前共有多少个可识别用户（在录入之前加一），亦即当前用户的编号 | 
note that when writing here, `Total_face_num` is the label of each picture, that is, how many identifiable users are currently in total (add one before entering), that is, the number of the current user

```python
 cv2.rectangle(img, (x, y), (x + w, y + w), (255, 0, 0))
 cv2.imwrite("./data/User." + str(T) + '.' + str(sample_num) + '.jpg', gray[y:y + h, x:x + w])
```
然后在循环末尾最后打印一个进度条，用于提示采集图像的进度 | then print a progress bar at the end of the loop to indicate the progress of image acquisition

主要原理就是每次输出不换行并且将光标移动到当前行的开头，输出内容根据进度不断变化即可，同时在控件的提示框也输出进度信息 | 
the main principle is that the output does not wrap each time and the cursor is moved to the beginning of the current line. The output content changes according to the progress, and the progress information is also output in the prompt box of the control

```python
print("\r" + "%{:.1f}".format(sample_num / pictur_num * 100) + "=" * l + "->" + "_" * r, end="")
var.set("%{:.1f}".format(sample_num / pictur_num * 100))  # 控件可视化进度信息 | control visualization progress information
window.update()  # 刷新控件以实时显示进度 | refresh controls to show progress in real time
```
#### 2.2训练识别器
读取data文件夹，读取照片内的信息，得到两个数组，一个faces存的是所有脸部信息、一个ids存的是faces内每一个脸部对应的标签，然后将这两个数组传给 `recog.train`用于训练 | 
read the data folder, read the information in the photo, and get two arrays, one faces stores all face information, and one ids stores the labels corresponding to each face in faces, and then transfer the two arrays Give `recog.train` for training

```python
    # 训练模型  #将输入的所有图片转成四维数组 | training model #Convert all the input pictures into a four-dimensional array
    recog.train(faces, np.array(ids))
```


训练完毕后保存训练得到的识别器到.yml文件中，文件名为人脸编号+.yml | 
after training, save the trained recognizer to the .yml file, and the file name is face number+.yml

```python
 recog.save(str(Total_face_num) + ".yml")
```

#### 2.3修改配置文件 | modify the configuration file
每一次训练结束都要修改配置文件，具体要修改的地方是第一行和最后一行。| 
the configuration file needs to be modified every time the training ends, and the specific places to be modified are the first line and the last line.

第一行有一个整数代表当前系统已经录入的人脸的总数，每次修改都加一。这里修改文件的方式是先读入内存，然后修改内存中的数据，最后写回文件。 | 
there is an integer on the first line representing the total number of faces that the system has entered, and one is added for each modification. The way to modify the file here is to read it into the memory first, then modify the data in the memory, and finally write it back to the file.

```python
    f = open('config.txt', 'r+')
    flist = f.readlines()
    flist[0] = str(int(flist[0]) + 1) + " \n"
    f.close()
    
    f = open('config.txt', 'w+')
    f.writelines(flist)
    f.close()

```
还要在最后一行加入一个二元组用以标识用户。| also add a two-tuple to the last line to identify the user.
格式为：标签+空格+用户名+空格，用户名默认为Userx（其中x标识用户编号）| the format is: label + space + user name + space, and the default user name is Userx 
(where x indicates the user number)

```python
f.write(str(T) + " User" + str(T) + " \n")
```

### 3.人脸识别（刷脸）| Face recognition 

<img src="https://gitee.com/Cheney822/images/raw/master/PicGo/20210622184401375.png"/>








由于这里采用多个`.yml`文件来储存识别器（实际操作时储存在一个文件中识别出错所以采用这种方式），所以在识别时需要遍历所有的.yml文件，如果每一个都不能识别才得出无法识别的结果，相反只要有一个可以识别当前对象就返回可以识别的结果。而对于每一个文件都识别十次人脸，若成功五次以上则表示最终结果为可以识别，否则表示当前文件无法识别这个人脸。| 
since multiple `.yml` files are used here to store the recognizer (the actual operation is stored in one file and the recognition error is adopted in this way), so it is necessary to traverse all the .yml files during recognition. If each one cannot be recognized Only then can an unrecognizable result be obtained. On the contrary, as long as there is one that can recognize the current object, a recognizable result will be returned. For each file, the face is recognized ten times. If it succeeds more than five times, it means that the final result can be recognized, otherwise it means that the current file cannot recognize the face.

识别过程中在GUI的控件中实时显示拍摄到的内容，并在人脸周围画一个矩形框，并根据识别器返回的结果实时显示在矩形框附近。 | 
during the recognition process, the captured content is displayed in the GUI control in real time, and a rectangular frame is drawn around the face, and is displayed near the rectangular frame in real time according to the result returned by the recognizer.

```python
idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
# 加载一个字体用于输出识别对象的信息 | load a font for outputting information about identifying objects
font = cv2.FONT_HERSHEY_SIMPLEX
# 输出检验结果以及用户名 | output the test result and username
cv2.putText(img, str(user_name), (x + 5, y - 5), font, 1, (0, 0, 255), 1)
cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (0, 0, 0), 1)
```





## 多线程 | multithreading：
程序的两个功能之间可以独立运行，就需要采用多线程的方法，但当遇到临界资源的使用时，多个进程/线程之间就要互斥的访问以免出错，本程序中具体的设计方法：
本程序采用多线程的方法实现并行。
程序的三个按钮对应着三个功能，分别是录入人脸、人脸检测、退出程序。
由于程序中的用户界面是利用python中的tkinter库做的，其按钮的响应函数用command指出，所以这里在每个`command`跳转到的函数中设置多线程，每敲击一次就用`threading.Thread`创建一个新的线程，然后在新的线程的处理函数`target`中实现按钮原本对应的功能 | 
the two functions of the program can run independently, so it is necessary to adopt a multi-threading method, but when encountering the use of critical resources, multiple processes/threads must mutually exclusive access to avoid errors. 
Design method:
This program uses multi-threaded methods to achieve parallelism.
The three buttons of the program correspond to three functions, which are face entry, face detection, and exit from the program.
Since the user interface in the program is made using the tkinter library in python, the response function of the button is pointed out by command, so here we set up multi-threading in the function that each `command` jumps to, and use ` threading.Thread`Creates a new thread, and then implements the original corresponding function of the button in the processing function `target` of the new thread.

```python
p = threading.Thread(target=f_scan_face_thread)
```

在涉及到摄像头的访问时，线程之间需要互斥的访问，所以设置了一个全局的变量`system_state_lock` 来表示当前系统的状态，用以实现带有优先级的互斥锁的功能。
锁状态为0表示摄像头未被使用，1表示正在刷脸，2表示正在录入新面容。
程序在实际执行的过程中如果状态为0，则无论是刷脸还是录入都能顺利执行，如果状态为1表示正在刷脸，如果此时敲击刷脸按钮则，系统会提示正在刷脸并拒绝新的请求，如果此时敲击录入面容按钮，由于录入面容优先级比刷脸高，所以原刷脸线程会被阻塞，| 
when it comes to camera access, mutually exclusive access is required between threads, so a global variable `system_state_lock` is set to represent the current state of the system to implement the function of a priority mutex.
A lock status of 0 means that the camera is not in use, 1 means that the face is being scanned, and 2 means that a new face is being recorded.
During the actual execution of the program, if the status is 0, it can be executed smoothly no matter whether it is facial recognition or input. If the status is 1, it means that the facial recognition is in progress. Reject the new request. If you tap the face entry button at this time, because the priority of face entry is higher than that of face brushing, the original thread of face brushing will be blocked.

```python
global system_state_lock
while system_state_lock == 2:  # 如果正在录入新面孔就阻塞 | block if new faces are being entered
	  pass
```

新的录入面容进程开始执行并修改系统状态为2，录入完成后状态变为原状态，被阻塞的刷脸进程继续执行，录入人脸线程刚执行完录入阶段现在正在训练，此时有两个线程并行，以此来保证训练数据的同时不影响系统的使用。| 
the new face entry process starts to execute and changes the system state to 2. After the entry is completed, the state changes to the original state, and the blocked face brushing process continues to execute. The face entry thread has just finished the entry stage and is now training. At this time, there are two Threads are parallelized to ensure that the training data does not affect the use of the system.


对于退出的功能，直接在函数内调用`exit()`，但是python的线程会默认等待子线程全部结束再退出，所以用`p.setDaemon(True)`将线程设置为守护线程，这样在主线程退出之后其它线程也都退出从而实现退出整个程序的功能。| 
for the exit function, call `exit()` directly in the function, but the python thread will wait for all sub-threads to end before exiting by default, so use `p.setDaemon(True)` to set the thread as a daemon thread, so that in the main After the thread exits, other threads also exit to realize the function of exiting the entire program.




## GUI设计 | GUI design：
程序采用python中的tkinter库做可视化，优点是占用资源小、轻量化、方便。| the program uses the tkinter library in python for visualization, which has the advantages of small resource occupation, light weight and convenience.

- 首先创建一个窗口命名为window然后设置其大小和标题等属性。| first create a window named window and then set its size and title properties.

- 然后在界面上设定一个绿底的标签，类似于一个提示窗口的作用 | then set a green label on the interface, which is similar to the function of a prompt window

- 然后分别创建三个按钮，并设置响应函数和提示字符，放置在window内部。| then create three buttons respectively, and set the response function and prompt character, and place them inside the window.

- 然后设置一个label类型的控件用于动态的展示摄像头的内容(将摄像头显示嵌入到控件中)。具体方法：创建video_loop()函数，在函数内访问全局的变量img，img是从摄像头读取到的图像数据。然后把img显示在label内。| 
then set a label type control to dynamically display the content of the camera (embed the camera display into the control). Specific method: create a video_loop() function, access the global variable img in the function, and img is the image data read from the camera. Then display the img in the label.

使用window.after方法，在给定时间后调用函数一次，实现固定时间刷新控件，从而达到实时显示摄像头画面在GUI中的效果。 | 
use the window.after method to call the function once afer a given time to refresh the control at a fixed time, so as to achieve the effect of real-time display of the camera image in the GUI.

```python
window.after(1, video_loop)
# 这句的意思是一秒以后执行video_loop函数 | this means to execute the video_loop function after one second
# 因为这一句是写在video_loop函数中的所以每过一秒函数执行一次。 | because this sentence is written in the video_loop function, the function is executed every second.
```
# 运行测试 | run test
## 说明 | illustrate
测试环境 | test enviroment：python 3.6 + opencv-python 3.4.14.51
需要的包 | package needed：

![在这里插入图片描述 | insert image description here](https://gitee.com/Cheney822/images/raw/master/PicGo/20210702213815631.png)



## 录入人脸 | enter face

从数据集录入 | import from dataset

![在这里插入图片描述 | insert image description here](https://gitee.com/Cheney822/images/raw/master/PicGo/2021070221383573.png)
从摄像头录入
![在这里插入图片描述 | insert image description here](https://gitee.com/Cheney822/images/raw/master/PicGo/20210702213904483.png)

![在这里插入图片描述 | insert image description here](https://gitee.com/Cheney822/images/raw/master/PicGo/20210702213915241.png)

## 人脸识别 | face recognition
![在这里插入图片描述 | insert image description here](https://gitee.com/Cheney822/images/raw/master/PicGo/20210702213930210.png)
![在这里插入图片描述 | insert image description here](https://gitee.com/Cheney822/images/raw/master/PicGo/20210702213935524.png)

