# HOFEC
手物特征增强互补模型（Hand-Object Feature Enhancement Complementary，HOFEC）是一个用于联合估计手物姿态的网络模型~

### 项目结构<br>
```
${ROOT}  
|-- data  
|   |-- HO3D
|   |   |-- data
|   |   |   |-- train
|   |   |   |   |-- ABF10
|   |   |   |   |-- ......
|   |   |   |-- evaluation
|   |   |   |-- train_segLable
|   |   |   |-- ho3d_train_data.json
|   |-- DEX_YCB
|   |   |-- data
|   |   |   |-- 20200709-subject-01
|   |   |   |-- ......
|   |   |   |-- object_render
|   |   |   |-- dex_ycb_s0_train_data.json
|   |   |   |-- dex_ycb_s0_test_data.json
```

#### [测试数据](https://drive.google.com/drive/folders/1cEbBMWmMuV5EE5v1_Lzs8NU_W00LJsA4?usp=sharing)<br>

#### 训练 & 测试<br>
##### HO3D
```
windows:
bash sh/train_ho3d.sh
bash sh/train_ho3d_test.sh

linux:
sh sh/train_ho3d.sh
sh sh/train_ho3d_test.sh
```
##### Dex-ycb
```
windows:
bash sh/train_dex-ycb.sh
bash sh/train_dex-ycb_test.sh

linux:
sh sh/train_dex-ycb.sh
sh sh/train_dex-ycb_test.sh
```

### 致谢<br>

+ [HandOccNet](https://github.com/namepllet/HandOccNet)
+ [Semi-Hand-Object](https://github.com/stevenlsw/Semi-Hand-Object)
+ [HFL-Net](https://github.com/lzfff12/HFL-Net)
