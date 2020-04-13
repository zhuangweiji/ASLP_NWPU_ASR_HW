# 作业 2：

## 从 kaldi/src/decoder/lattice-faster-decoder.cc 中查找

### 1） histogram pruning 的代码段
 728   BaseFloat cur_cutoff = GetCutoff(final_toks, &tok_cnt, &adaptive_beam, &best_elem);
 通过GetCutoff()获取histogram的剪枝阈值，GetCutoff根据max_active, min_active来控制histogram pruning保留的token数

 774     if (tok->tot_cost <= cur_cutoff)
 在774行中，通过比较token的cost值和histogram pruning计算出的cur_cutoff来判断是否剪枝
 
### 2） beam pruning 的代码段
 741   // First process the best token to get a hopefully
 742   // reasonably tight bound on the next cutoff.  The only
 743   // products of the next block are "next_cutoff" and "cost_offset".
 744   if (best_elem) {
 745     StateId state = best_elem->key;
 746     Token *tok = best_elem->val;
 747     cost_offset = - tok->tot_cost;
 748     for (fst::ArcIterator<FST> aiter(*fst_, state);
 749          !aiter.Done();
 750          aiter.Next()) {
 751       const Arc &arc = aiter.Value();
 752       if (arc.ilabel != 0) {  // propagate..
 753         BaseFloat new_weight = arc.weight.Value() + cost_offset -
 754             decodable->LogLikelihood(frame, arc.ilabel) + tok->tot_cost;
 755         if (new_weight + adaptive_beam < next_cutoff)
 756           next_cutoff = new_weight + adaptive_beam;
 757       }
 758     }
 759   }
 
 在728行GetCutoff函数中我们同时算出了best_token与best路径之间的距离adaptive_beam，  
 在744行-759行之间，通过对best_token  进行传播，配合adaptive_beam找到令牌传递过程中  
 产生的最好阈值，获得beam pruning的阈值next_cutoff（这个传播不是真正的传播，不增加token）
 
 785           if (tot_cost >= next_cutoff) continue;
 786           else if (tot_cost + adaptive_beam < next_cutoff)
 787             next_cutoff = tot_cost + adaptive_beam; // prune by best current token
 在785-787行中，查找最佳next_cutoff
 785行在查找时使用当前最佳阈值做了一个小剪枝
 786、787行当更好的阈值出现时，动态更新beam pruning阈值
 
 870         if (tot_cost < cutoff) 
 最终在ProcessNonemittin()870行进行beam pruning。
 