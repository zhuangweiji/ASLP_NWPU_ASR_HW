#作业一：

#0)导入环境变量
#cd ~zhuangweiji/git/kaldi-master-cuda10.1/egs/wsj/s5/
#. path.sh 
#cd -

#1)将图示a)和b)两个WFST写成text格式a.txt.fst和b.txt.fst。
cat > a.text.fst << EOF
0	1	a	b	0.1
0	2	b	a	0.2
1	1	c	a	0.3
1	3	a	a	0.4
2	3	b	b	0.5
3	0.6
EOF
cat > b.text.fst << EOF
0	1	b	c	0.3
1	2	a	b	0.4
2	2	a	b	0.6
2	0.7
EOF

#2)定义input label和output label的字符表(即字符到数值的映射)。
cat > input_label.txt << EOF
<eps> 0
a 1
b 2
c 3
EOF
cat > output_label.txt << EOF
<eps> 0
a 1
b 2
c 3
EOF

#3)生成a)和b)对应的binary格式WFST,记为a.fst和b.fst。
fstcompile --isymbols=input_label.txt --osymbols=output_label.txt a.text.fst a.fst
fstcompile --isymbols=input_label.txt --osymbols=output_label.txt b.text.fst b.fst

#4)进行compose，并输出out.fst。
fstcompose a.fst b.fst compose.fst

#5)打印输出的样子。
fstdraw --portrait --isymbols=input_label.txt --osymbols=output_label.txt compose.fst compose.dot
dot -Tpng compose.dot -o compose.png
