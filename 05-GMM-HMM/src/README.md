## 1、思考题
回答lab2.txt中的问题，见../lab2.txt

## 2、编程作业
1. 理解数据文件
2. ref文件作为参考输出，用diff命令检查自己的实现得到的输出和ref是否完全一致
3. 实验中实际用的GMM其实都是单高斯
4. 阅读util.h里面的注释，Graph的注释有如何遍历graph中state上所有的arc的方法。

### 2.0 CentOS服务器上安装开发环境、boost库，给shell脚本加执行权限
yum -y install make g++ libboost-all-dev boost boost-devel boost-doc
chmod +x ./*.sh

### 2.1 作业p1：viterbi解码
用viterbi解码，完成lab2_vit.C中的一处代码。
#### 运行:
make -C src
./lab2_p1a.sh
./lab2_p1b.sh
#### 比较结果: 
vimdiff p1a.chart p1a.chart.ref

### 2.2 作业p2：viterbi-EM
给定align（viterbi解码的最优状态(或边）序列)，原始语音和GMM的初始值，
用viterbi解码得到的最优的一条序列来计算统计量，估计模型参数，更新GMM参数。
完成src/gmm_util.C中两处代码。
#### 运行：
make -C src
./lab2_p2a.sh
#### 比较结果：
vimdiff p2a.gmm p2a.gmm.ref

### 2.3 作业p3：forward-backward-EM
用前向后向算法计算统计量，估计模型参数，完成src/lab2_fb.C中的两处代码。
#### 运行：
./lab2_p3a.sh  # 1条数据，1轮迭代
./lab2_p3b.sh  # 22条数据，1轮迭代
./lab2_p3c.sh  # 22条数据，20轮迭代
./lab2_p3d.sh  # 用viterbi算法解码p3c的训练的模型，结果应该和p1b的结果一样
#### 比较结果:
vimdiff p3a_chart.dat p3a_chart.ref
vimdiff p3b_chart.dat p3a_chart.ref
vimdiff p3b.gmm p3b.gmm.ref。
