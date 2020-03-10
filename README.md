# 《语音识别：从入门到精通》 课程作业
You can find homework file for the asr course from here

## 01-OUTLINE

## 02-feature-extraction
给定一段音频，请提取12维MFCC特征和23维FBank，阅读代码预加重、分帧、加窗部分，完善作业代码中FBank特征提取和MFCC特征提取部分。
给出最终的FBank特征和MFCC特征，存储在纯文本中，用默认的配置参数，无需进行修改。

## 03-GMM-EM
本次实验所用的数据为0-9（其中0的标签为Z（Zero））和o这11个字符的英文录音，每个录音的原始录音文件和39维的MFCC特征都已经提供，
实验中，每个字符用一个GMM来建模，在测试阶段，对于某句话，对数似然最大的模型对应的字符为当前语音数据的预测的标签（target）。

## 04-HMM
1、用python编程实现前向算法和后向算法。
2、用python编程实现viterbi算法，求最优序列。

## 05-GMM-HMM
作业来源：哥伦比亚大学语音识别课程E6870
难度系数：9 
程序设计语言：C++ 
预计花费时间：5~10个小时
• 作业内容：
  • Viterbi解码
  • 估计GMM参数
  • 前向后向训练，利用前向后向训练估计GMM参数 

## 06-基于DNN-HMM的语音识别系统作业
### 作业1
本实验实现了一个简单的DNN的框架，使用DNN进行11个数字的训练和识别。
实验中使用训练和测试数据分别对该DNN进行训练和测试。
阅读dnn.py中的代码，理解该DNN框架，完善ReLU激活函数和FullyConnect全连接层的前向后向算法。
运行该实验，该程序末尾会打印出在测试集上的准确率。假设实现正确，应该得到95%以上的准确率。
除了跑默认参数之外，读者还可以自己尝试调节一些超参数，并观察这些超参数对最终准确率的影响。如
* 学习率 / 隐层结点数 / 隐层层数  
读者还可以基于该框架实现神经网络中的一些基本算法，如：
* sigmoid和tanh激活函数 / dropout / L2 regularization / optimizer(Momentum/Adam)  
实现后读者可以在该数字识别任务上应用这些算法，并观察对识别率的影响。
通过调节这些超参数和实现其他的一些基本算法，读者可以进一步认识和理解神经网络。
### 作业2
基于Kaldi理解基于DNN-HMM的语音识别系统。请安装kaldi，并运行kaldi下的标准数据集THCHS30的实验，

## 07-LM
根据"实验指导书.pdf":part2.2和part4分别完成N-gram计数和Witten-Bell算法的编写。

## 08-WFST-decoder
### 作业1
参考openfst官网，根据图示写出WFST，并将其compose，画出结果
### 作业2
查看kaldi lattice-faster-decoder的剪枝算法
### 作业3
运行kaldi/egs/mini_libispeech，训练模型，报告wer

## 09-DT-LFMMI

## 10-E2E

## 11-SUMMARY
