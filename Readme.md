数据集
========
下载地址：
https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip

**记得下载完数据之后将数据Data文件夹放在model.py的同级目录下。**

配置文档
=========
**python版本：python2**<br>

相关依赖包：
-------------
**h5py 2.10.0**<br>
**keras 2.0.3**<br>
**numpy 1.16.6**<br>
**haversine 2.2.0**<br>


使用文档
=========
**在model.py同级目录下建立stf_rnn_save_models目录用来保存模型**<br>
运行训练集：python STF-RNN_train.py<br>
运行验证集：python STF-RNN_test.py<br>
