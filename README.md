# 公共场所吸烟检测与EasyEdge部署
&emsp;&emsp;公共场所进行吸烟检测，如果发现有吸烟行为，及时警告并记录。

## 一、项目背景

&emsp;&emsp;**吸烟有害健康**。

&emsp;&emsp;为减少和消除烟草烟雾危害，保障公众健康，根据国务院立法工作计划，2013年卫生计生委启动了《公共场所控制吸烟条例》起草工作。按照立法程序的有关要求，在总结地方控烟工作经验的基础上，深入调研，广泛征求了工业和信息化部、烟草局等25个部门，各省级卫生计生行政部门、部分行业协会及有关专家的意见，经不断修改完善，形成了《公共场所控制吸烟条例（送审稿）》。送审稿明确，所有室内公共场所一律禁止吸烟。此外，体育、健身场馆的室外观众坐席、赛场区域；公共交通工具的室外等候区域等也全面禁止吸烟。 

&emsp;&emsp;但，仍存在公共场合吸烟问题，为此一种无人化、智能化吸烟检测装置的需求迫在眉睫。

![](https://ai-studio-static-online.cdn.bcebos.com/ef0e7a73e61e488ba4a44e518bd8a54c3de08fc60c744347a8370c8e2d759695)


## 二、数据集简介

&emsp;&emsp;本次数据集从浆友公开数据集中获取。

&emsp;&emsp;具体链接为：https://aistudio.baidu.com/aistudio/datasetdetail/94796。

&emsp;&emsp;此处可细分，如下所示：


本项目使用的吸烟检测数据集已经按VOC格式进行标注，目录情况如下：
```
 dataset/                        
 ├── annotations/    
 ├── images/       
 ```

![](https://ai-studio-static-online.cdn.bcebos.com/cc05176c3deb4f3fa048d0803c500ce993fda28a7c434cedbe38d14e30af8b25)


## 三、模块导入

&emsp;&emsp;PaddleX。

&emsp;&emsp;项目环境：Paddle 2.1.0
```
!pip install paddlex==1.3.11
!pip install paddle2onnx
```

## 四、解压数据集
```
# 进行数据集解压
!unzip -oq /home/aistudio/data/data102810/pp_smoke.zip -d /home/aistudio/dataset
```

## 五、数据处理和数据清洗
```
# 这里修改.xml文件中的<path>元素
!mkdir dataset/Annotations1
import xml.dom.minidom
import os

path = r'dataset/Annotations'  # xml文件存放路径
sv_path = r'dataset/Annotations1'  # 修改后的xml文件存放路径
files = os.listdir(path)
cnt = 1

for xmlFile in files:
    dom = xml.dom.minidom.parse(os.path.join(path, xmlFile))  # 打开xml文件，送到dom解析
    root = dom.documentElement  # 得到文档元素对象
    item = root.getElementsByTagName('path')  # 获取path这一node名字及相关属性值
    for i in item:
        i.firstChild.data = '/home/aistudio/dataset/JPEGImages/' + str(cnt).zfill(6) + '.jpg'  # xml文件对应的图片路径

    with open(os.path.join(sv_path, xmlFile), 'w') as fh:
        dom.writexml(fh)
    cnt += 1
```

```
# 这里修改.xml文件中的<failname>元素
!mkdir dataset/Annotations2
import xml.dom.minidom
import os

path = r'dataset/Annotations1'  # xml文件存放路径
sv_path = r'dataset/Annotations2'  # 修改后的xml文件存放路径
files = os.listdir(path)

for xmlFile in files:
    dom = xml.dom.minidom.parse(os.path.join(path, xmlFile))  # 打开xml文件，送到dom解析
    root = dom.documentElement  # 得到文档元素对象
    names = root.getElementsByTagName('filename')
    a, b = os.path.splitext(xmlFile)  # 分离出文件名a
    for n in names:
        n.firstChild.data = a + '.jpg'
    with open(os.path.join(sv_path, xmlFile), 'w') as fh:
        dom.writexml(fh)
```

```
# 这里修改.xml文件中的<name>元素
!mkdir dataset/Annotations3
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
 
import os
import xml.etree.ElementTree as ET
 
origin_ann_dir = '/home/aistudio/dataset/Annotations2/'# 设置原始标签路径为 Annos
new_ann_dir = '/home/aistudio/dataset/Annotations3/'# 设置新标签路径 Annotations
for dirpaths, dirnames, filenames in os.walk(origin_ann_dir):   # os.walk游走遍历目录名
  for filename in filenames:
    # if os.path.isfile(r'%s%s' %(origin_ann_dir, filename)):   # 获取原始xml文件绝对路径，isfile()检测是否为文件 isdir检测是否为目录
    origin_ann_path = os.path.join(origin_ann_dir, filename)   # 如果是，获取绝对路径（重复代码）
    new_ann_path = os.path.join(new_ann_dir, filename)
    tree = ET.parse(origin_ann_path)  # ET是一个xml文件解析库，ET.parse（）打开xml文件。parse--"解析"
    root = tree.getroot()   # 获取根节点
    for object in root.findall('object'):   # 找到根节点下所有“object”节点
        name = str(object.find('name').text)  # 找到object节点下name子节点的值（字符串）
        # 如果name等于str，则删除该节点
        if (name in ["smoke"]):
        #   root.remove(object)
            pass

        # 如果name等于str，则修改name
        else:
            object.find('name').text = "smoke"

        tree.write(new_ann_path)#tree为文件，write写入新的文件中。
```
```
#删除冗余文件并修改文件夹名字
!rm -rf dataset/Annotations
!rm -rf dataset/Annotations1
!rm -rf dataset/Annotations2
!mv dataset/Annotations3 dataset/Annotations
!mv dataset/images dataset/JPEGImages

```
```
#在原始数据集中，存在.jpg文件和.xml文件匹配不对等的情况，这里我们根据.jpg文件名删除了在Annotations文件夹中无法匹配的.xml文件，
#使得.jpg和.xml能够一一对应。
import os
import shutil
path_annotations = 'dataset/Annotations'
path_JPEGImage = 'dataset/JPEGImages'
xml_path = os.listdir(path_annotations)
jpg_path = os.listdir(path_JPEGImage)
for i in jpg_path:
    a = i.split('.')[0] + '.xml'
    if a in xml_path:
        pass
    else:
        print(i)
        os.remove(os.path.join(path_JPEGImage,i))

```
```
#划分数据集
#基于PaddleX 自带的划分数据集的命令，数据集中训练集、验证集、测试集的比例为7:2:1。
!paddlex --split_dataset --format VOC --dataset_dir /home/aistudio/dataset/ --val_value 0.2 --test_value 0.1

```

## 六、模型训练
```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from paddlex.det import transforms
import paddlex as pdx


# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=350), transforms.RandomDistort(),
    transforms.RandomExpand(), transforms.RandomCrop(), transforms.Resize(
        target_size=608, interp='RANDOM'), transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.Resize(
        target_size=608, interp='CUBIC'), transforms.Normalize()
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
train_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/aistudio/dataset',
    file_list='/home/aistudio/dataset/train_list.txt',
    label_list='/home/aistudio/dataset/labels.txt',
    transforms=train_transforms,
    parallel_method='thread',
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/aistudio/dataset',
    file_list='/home/aistudio/dataset/val_list.txt',
    label_list='/home/aistudio/dataset/labels.txt',
    parallel_method='thread',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
num_classes = len(train_dataset.labels)

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-ppyolo
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='MobileNetV3_large')


# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#train
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=300,
    train_dataset=train_dataset,
    train_batch_size=24,
    eval_dataset=eval_dataset,
    learning_rate=0.001 / 8,
    warmup_steps=1000,
    warmup_start_lr=0.0,
    save_interval_epochs=1,
    lr_decay_epochs=[240, 270],
    use_vdl= True,
    save_dir='output/yolov3_mobilenet')
```
```
可视化
!visualdl --logdir home/aistudio/output/yolov3_mobilenet/vdl_log --port 8001
```

## 七、模型评估
```
model.evaluate(eval_dataset, batch_size=1, epoch_id=None, return_details=False)

```

## 八、模型导出
```
#把模型导出，下载本地，然后上传到EasyEdge
!paddlex --export_inference --model_dir=/home/aistudio/output/yolov3_mobilenet/best_model --save_dir=./down_model
```

## 九、模型送到[EasyEdge](https://ai.baidu.com/easyedge/home)里面，部署APP与Window桌面应用

&emsp;&emsp;EasyEdge是基于百度飞桨轻量化推理框架Paddle Lite研发的端与边缘AI服务平台，能够帮助深度学习开发者将自建模型快速部署到设备端。只需上传模型，最快2分种即可获得适配终端硬件/芯片的模型。
![](https://ai-studio-static-online.cdn.bcebos.com/d8ed555a53cb49aabe9291586fe7fef9f5887abf948a44aba50ce15cb5d91176)

#只需要把导出的模型分别对应上传到EasyEdge，labels.txt这个文件就是之前数据集里面的标签文件

【第一步】
![](https://ai-studio-static-online.cdn.bcebos.com/0c76b12dc38545789b0f770e7770a1925a54664ba0284785ad5ca823a12cc7ac)

【第二步】
![](https://ai-studio-static-online.cdn.bcebos.com/8343dcc87e4b43c58d37cf9074b1ce45c975fc017b8f42f9a97608b28a1b9edf)

【第三步】
![](https://ai-studio-static-online.cdn.bcebos.com/929778a0becd48539f498bb96d757f916e25073287c240acad9a2ee2115b8c26)

【第四步】
![](https://ai-studio-static-online.cdn.bcebos.com/ea99ae519ad94805b51847d7a31061d5effd214199a544a0900bf2e2a6f842a9)

【第五步】
![](https://ai-studio-static-online.cdn.bcebos.com/c4ca64681be049498bb5e30874041f1864f5f6111f7248e393601974fcb0cbce)

【第六步】
![](https://ai-studio-static-online.cdn.bcebos.com/2761fce7eefe4e09b95627f5f8282fdce8ea8dfdc6fb4b12b8b09cae58789efb)

【第七步】
![](https://ai-studio-static-online.cdn.bcebos.com/83568c54ee1140c29e8f68dc5bdd7e75eb9a023a9ac7457da52e3f38258173e1)

【第八步】
![](https://ai-studio-static-online.cdn.bcebos.com/6f6858555967467fba711eb4466fbeea208b2c1203c8464cba6ad8f630d7cc70)

【第九步】
![](https://ai-studio-static-online.cdn.bcebos.com/d92b94469260430ea121407e290be0a41046f33186c7485dbcb19b245d9010b6)

【第十步】
![](https://ai-studio-static-online.cdn.bcebos.com/e62b088187ee4d43be46e56901fe061e3507ecf97c1a41fca46f07dbe806cd01)

【第十一步】
![](https://ai-studio-static-online.cdn.bcebos.com/afabdc56db1343ee9270d0c4c223d69bf20cfbb005ca4e49b191007df33fb979)

APP端同样，按操作就可以了

