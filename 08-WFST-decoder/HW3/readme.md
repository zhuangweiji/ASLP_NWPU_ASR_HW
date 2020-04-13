## 作业 3：  

### 0）运行 kaldi/egs/mini_librispeech 至少训练完 3 音素模型 tri1.  
cd ~/git/kaldi-master-cuda10.1/egs/mini_librispeech/s5  
./run.sh  
  
### 1）此时你的 data/lang_nosp_test_tglarge 中无 G.fst 文件，将 data/local/lm/lm_tglarge.arpa.gz转化为 G.fst 存于其中。[提交你的完整命令]  
gunzip -c data/local/lm/lm_tglarge.arpa.gz |arpa2fst --disambig-symbol=#0 --read-symbol-table=data/lang_nosp_test_tglarge/words.txt - data/lang_nosp_test_tglarge/G.fst   

### 2）用 tri1 模型和 tgsmall 构建的 HCLG 图解码 dev_clean_2 集合的“1272-135031-0009”句，输出 Lattice 和 CompactLattice 的文本格式。[提交你的完整命令和输出文件]  
cd exp/tri1/decode_nosp_tgsmall_dev_clean_2  
gunzip  lat.1.gz  
echo "1272-135031-0009 lat.1:853636" |lattice-copy --write-compact=false scp:- ark,t:lattice_normal.txt     
echo "1272-135031-0009 lat.1:853636" |lattice-copy --write-compact=true scp:- ark,t:lattice_compact.txt       
cd -  

### 3）使用 1)中生成的 tglarge 的 G.fst 和 steps/lmrescore.sh 对exp/tri1/decode_nosp_tgsmall_dev_clean_2 中的 lattice 重打分，汇报 wer。  
steps/lmrescore.sh --cmd "utils/run.pl" data/lang_nosp_test_tgsmall data/lang_nosp_test_tglarge data/dev_clean_2 exp/tri1/decode_nosp_tgsmall_dev_clean_2 exp/tri1/decode_nosp_tglarge_dev_clean_2  
grep WER exp/tri1/decode_nosp_tglarge_dev_clean_2/wer* |utils/best_wer.sh  
#结果：  
#%WER 20.98 [ 4224 / 20138, 457 ins, 591 del, 3176 sub ] exp/tri1/decode_nosp_tglarge_dev_clean_2/wer_15_0.  