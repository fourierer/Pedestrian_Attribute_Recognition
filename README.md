# Pedestrian_Attribute_Recognition
Recognizing pedestrian attributes on RAP(V2) via pytorch, copy from 'https://github.com/dangweili/pedestrian-attribute-recognition-pytorch' ,annotate the code and the learn model 



再次强调，这个仓库中的代码是从https://github.com/dangweili/pedestrian-attribute-recognition-pytorch 搬运来的，代码没有原创性。因为课程作业需要跑通这个行人属性的算法，所以这里记录该问题的学习过程以及运行代码过程的遇到的问题。



相关文章：A Richly Annotated Dataset for Pedestrian Attribute Recognition, ACPR 2015; A Richly Annotated Pedestrian Dataset for Person Retrieval in Real Surveillance Scenario, TIP 2018

1.数据集RAP(V2)

（1）下载主页：http://www.rapdataset.com/ RAP共有41,585个行人样本，每个样本都带有72个属性，数据下载后有数据集文件夹RAP_dataset和数据集注释文件夹RAP_annotation。

（2）RAP_dataset中存放数据集，每个样本是一个行人，如下：

![CAM01_2014-02-15_20140215161032-20140215162620_tarid38_frame2590_line1](/Users/momo/Documents/视频处理与分析/Pedestrian_Attribute_Recognition/RAP/RAP_dataset/CAM01_2014-02-15_20140215161032-20140215162620_tarid38_frame2590_line1.png)

（3）RAP_annotation文件夹中主要有RAP_annotation.mat，用来记录标签信息。ReadMe.txt中有简短的说明，RAP_annotation.mat文件具体说明如下：

文件中共有7个字段，分别是：imagesname, position, label, partion, attribute_chinese, attribute_eng, attribute_exp.

1）RAP_annotation.imagesname is the 41585 image names in the RAP dataset and the image could be download in another zip file in the dataset website.数据集中图片的名字，共41585张图片，41585*1cell；

2）RAP_annotation.position is the absolute coordinate of the person bounding box of fullbody, head-shoulder, upperbody, lowerbody in the full image.
Each of bounding box has four points, including (x,y,w,h). The coordinates are indexed from zero.If the part is invisibile, the corresponding coordinate are set to be are zero, such as (0, 0, 0, 0).位置是全图中满体、头肩、上体、下体的人包围框的绝对坐标。每个边界框有四个点，包括(x,y,w,h)。41585*16int32；

3）The attributes' name in english and chinese are shown in RAP_annotation.attribute_eng and RAP_annotation.attribute_chinese.
The "hs", "ub" and "lb" in attribute_eng mean head-shoulder, upperbody and lowerbody respectly.样本的92个属性，包括中英文，92*1cell；

4）partion:RAP_annotation.partion is the 5 random partion for training and test, which is the same as the setting in our paper.是所有数据集中训练集和测试集的5个随机划分，每个划分里面的训练集和测试集中有图片的id；

5）RAP_annotation.attribute_exp is the shorted name of top 51 attributes in attribute_eng, which is the same as our paper.英文属性中最短的前51个属性；

6）RAP_annotation.label is the annotation for the attributes in RAP_annotation.attribute_eng. Each row in RAP_annotation.label is an example, which is corresponding with each image in RAP_annotation.imagesname.41585*92，给每个样本标注了92个属性，每个属性的标签是0或者1；



2.数据集预处理生成文件索引

下载的repository中有好几个dataset文件夹，先将下载好的数据集放在与script同目录下的dataset/rap文件夹当中（不同数据集放入该dataset不同的文件夹中，如原repository中的peta_release数据集放到peta文件夹中），在当前目录下执行：

```shell
python script/dataset/transform_rap.py
```





3.训练



4.测试



