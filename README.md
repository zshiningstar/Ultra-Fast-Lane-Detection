### 环境安装
#### 1 conda安装
> [参考](https://github.com/zshiningstar/yolov5/blob/mydata/README)
#### 2 虚拟环境安装:官方推荐使用python3.8,因此使用conda新建一个python3.8环境
 - 环境创建
```
 # 新建环境
conda create -n lane python=3.8
 # 进入环境
conda activate lane
 # 安装pytorch 和 cuda 
 # 不需要翻墙
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 
```
#### 3 源码下载
```
git clone https://github.com/zshiningstar/Ultra-Fast-Lane-Detection.git
```
#### 4 源码依赖安装
```
conda activate lane
pip install -r requirements.txt
```
 - 使用conda安装独立的ros python
   在python下调用ROS相关接口时,使用apt安装的通常仅集成在python2,兼容方面很多问题,为了构造环境相对干净,与系统环境隔绝的运行环境,使用conda安装ros python
```
conda install -c conda-forge ros-rospy
```
### 5 训练好的模型文件下载
链接: https://pan.baidu.com/s/1jiVB4J_4173OXXjAzp5qTA 提取码: 49ee
下载后放在model_pth必须放在文件夹下

#### 6 程序启动
 - 方法1:
```
python lane_detect_ros_node.py configs/ros_config.py  --test_model model_pth/ep049.pth
```
 - 方法2:
```
sh launch/lane_detect_ros_node.sh
```
#### 7 ros接口
 - lane_detect_ros_node.py文件为ros接口
 - 输入话题消息:**/image_raw**
 - 发布结果话题消息:**/lane_detected**
