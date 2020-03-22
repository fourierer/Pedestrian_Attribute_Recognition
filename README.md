# Pedestrian_Attribute_Recognition
Recognizing pedestrian attributes on RAP(V2) via pytorch, copy from 'https://github.com/dangweili/pedestrian-attribute-recognition-pytorch' ,annotate the code and the learn model 



再次强调，这个仓库中的代码是从https://github.com/dangweili/pedestrian-attribute-recognition-pytorch 搬运来的，代码没有原创性。因为课程作业需要跑通这个行人属性的算法，所以这里记录该问题的学习过程以及运行代码过程的遇到的问题。



相关文章：A Richly Annotated Dataset for Pedestrian Attribute Recognition, ACPR 2015; A Richly Annotated Pedestrian Dataset for Person Retrieval in Real Surveillance Scenario, TIP 2018



先给出原本github上数据集peta_relase的训练结果：

![result_of_peta](/Users/momo/Documents/Pedestrian_Attribute_Recognition/result_of_peta.png)



1.数据集RAP

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

下载的repository中有好几个dataset文件夹，先将下载好的数据集放在与script同目录下的dataset/rap文件夹当中（不同数据集放入该dataset不同的文件夹中，如原repository中的peta_release数据集放到peta文件夹中），在当前目录下执行下面语句，生成数据集划分索引和数据集标签索引分别于/pedestrian-attribute-recognition-pytorch/dataset/rap/rap_partition.pkl(rap_dataset.pkl)：

```shell
python script/dataset/transform_rap.py
```

transform_rap.py:

```python
import os
import numpy as np
import random
import cPickle as pickle
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """
    dataset = dict()
    dataset['description'] = 'rap'
    dataset['root'] = './dataset/rap/RAP_dataset/'
    dataset['image'] = []
    dataset['att'] = []
    dataset['att_name'] = []
    dataset['selected_attribute'] = range(51)
    # load Rap_annotation.mat
    data = loadmat(open('./dataset/rap/RAP_annotation/RAP_annotation.mat', 'r'))
    for idx in range(51):
        dataset['att_name'].append(data['RAP_annotation'][0][0][6][idx][0][0])

    for idx in range(41585):
        dataset['image'].append(data['RAP_annotation'][0][0][5][idx][0][0])
        dataset['att'].append(data['RAP_annotation'][0][0][1][idx, :].tolist())

    with open(os.path.join(save_dir, 'rap_dataset.pkl'), 'w+') as f:
        pickle.dump(dataset, f)

def create_trainvaltest_split(traintest_split_file):
    """
    create a dataset split file, which consists of index of the train/val/test splits
    """
    partition = dict()
    partition['trainval'] = []
    partition['test'] = []
    partition['weight_trainval'] = []
    # load RAP_annotation.mat
    data = loadmat(open('./dataset/rap/RAP_annotation/RAP_annotation.mat', 'r'))
    for idx in range(5):
        trainval = (data['RAP_annotation'][0][0][0][idx][0][0][0][0][0,:]-1).tolist()
        test = (data['RAP_annotation'][0][0][0][idx][0][0][0][1][0,:]-1).tolist()
        partition['trainval'].append(trainval)
        partition['test'].append(test)
        # weight
        weight_trainval = np.mean(data['RAP_annotation'][0][0][1][trainval, :].astype('float32')==1, axis=0).tolist()
        partition['weight_trainval'].append(weight_trainval)
    with open(traintest_split_file, 'w+') as f:
        pickle.dump(partition, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="rap dataset")
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./dataset/rap/')
    parser.add_argument(
        '--traintest_split_file',
        type=str,
        default="./dataset/rap/rap_partition.pkl")
    args = parser.parse_args()
    save_dir = args.save_dir
    traintest_split_file = args.traintest_split_file

    generate_data_description(save_dir)
    create_trainvaltest_split(traintest_split_file)
```

代码测试：

（1）dataset['att_name']，输出51个名字最短的属性

```python
print dataset['att_name']
```

[u'Female', u'AgeLess16', u'Age17-30', u'Age31-45', u'BodyFat', u'BodyNormal', u'BodyThin', u'Customer', u'Clerk', u'BaldHead', u'LongHair', u'BlackHair', u'Hat', u'Glasses', u'Muffler', u'Shirt', u'Sweater', u'Vest', u'TShirt', u'Cotton', u'Jacket', u'Suit-Up', u'Tight', u'ShortSleeve', u'LongTrousers', u'Skirt', u'ShortSkirt', u'Dress', u'Jeans', u'TightTrousers', u'LeatherShoes', u'SportShoes', u'Boots', u'ClothShoes', u'CasualShoes', u'Backpack', u'SSBag', u'HandBag', u'Box', u'PlasticBag', u'PaperBag', u'HandTrunk', u'OtherAttchment', u'Calling', u'Talking', u'Gathering', u'Holding', u'Pusing', u'Pulling', u'CarryingbyArm', u'CarryingbyHand']



（2）dataset['image']，输出41585个图片文件的名字

```python
print dataset['image']
```

......u'CAM01_2014-02-15_20140215161032-20140215162620_tarid139_frame8486_line1.png', u'CAM01_2014-02-15_20140215161032-20140215162620_tarid136_frame8181_line1.png', u'CAM01_2014-02-15_20140215161032-20140215162620_tarid135_frame8121_line1.png', u'CAM01_2014-02-15_20140215161032-20140215162620_tarid128_frame7794_line1.png', u'CAM01_2014-02-15_20140215161032-20140215162620_tarid124_frame7735_line1.png', u'CAM01_2014-02-15_20140215161032-20140215162620_tarid123_frame7362_line1.png', u'CAM01_2014-02-15_20140215161032-20140215162620_tarid119_frame6898_line1.png', u'CAM01_2014-02-15_20140215161032-20140215162620_tarid109_frame6614_line1.png', u'CAM01_2014-02-15_20140215161032-20140215162620_tarid104_frame6583_line1.png', u'CAM01_2014-02-15_20140215161032-20140215162620_tarid101_frame6541_line1.png', u'CAM01_2014-02-15_20140215161032-20140215162620_tarid0_frame218_line1.png']

（3）dataset['att']，输出41585样本的92个属性标签信息，0-1标签

```python
print dataset['att']
```

......[1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]]

（4）dataset信息存入rap_dataset.pkl中

```python
with open(os.path.join(save_dir, 'rap_dataset.pkl'), 'w+') as f:
        pickle.dump(dataset, f)
```



（5）trainval和test，划分的训练集和测试集的序号

（6）partition['trainval']和partition['test']，不同划分下（5次随机划分）训练集与测试集序号



3.训练

```shell
sh script/experiment/train.sh
```

根据需要修改train.sh中的参数即可，代码实际上就是用ResNet-50做了多类别分类。



4.测试

```shell
sh script/experiment/test.sh
```



5.Demo

```shell
python script/experiment/demo.py
```

