# 导入KALDI_ROOT及所需env
cd ~/git/kaldi-master-cuda10.1/egs/wsj/s5
. path.sh
LC_ALL=

# 1.使用SRILM获得经过inerpolation式的Kneser-Ney平滑的3-gram以上的语言模型。总结你观察到的一些现象。
cat ~/git/kaldi-master-cuda10.1/egs/thchs30/NWPU-06-DNN-HMM/data/train/text |cut -d' ' -f2- >train.txt
ngram-count -interpolate -kndiscount -order 3 -text train.txt -lm thchs30.arpa
# 报错
# one of required modified KneserNey count-of-counts is zero
# error in discount estimator for order 3
# 数据集太小，部分count为零，discount为负，导致错误退出
# 使用默认平滑算法Written bell
ngram-count -interpolate -kndiscount -order 3 -text train.txt -lm thchs30.arpa


#2.使用SRILM计算在识别测试集上计算PPL。
cat ~/git/kaldi-master-cuda10.1/egs/thchs30/NWPU-06-DNN-HMM/data/test/text |cut -d' ' -f2- >test.txt
ngram -ppl test.txt -order 3 -lm thchs30.arpa 
# file test.txt: 2495 sentences, 49085 words, 18731 OOVs
# 17482 zeroprobs, logprob= -42184.13 ppl= 556.047 ppl1= 1893.219
ngram -ppl test.txt -order 3 -lm thchs30.arpa  -debug 1 &> test.ppl
