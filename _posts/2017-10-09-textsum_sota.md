---
layout: post
title: Abstractive Text Summarization State-of-the-Art
subtitle: A review of last models
---
<script type="text/x-mathjax-config">
        MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "none" } } }); 
</script>
# A. See (2017) -- Get to the point
*[[1704.04368] Abigail See, Peter J. Liu, Christopher D. Manning, (2017) Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*
* Dataset: CNN/DM 
* Implementation: <https://github.com/abisee/pointer-generator/>
* Results:    
  ![Abisee (2017) results](/img/abisee_2017_results.png)

### Model 
* Attention over source: 
  * Attention scores: \\[  e_i^t = v^T tanh(W_h h_i + W_s s_t + b_{attn}) \\]
  * Normalized scores: \\[  a^t = softmax(e^t) \\]
  * Context vectors: \\[ h_t^* = \sum_i{a_i^th_i} \\]
  * Probabilities over the vocabulary: \\[ P_{vocab}= softmax(V'(V[s_t, h_t^*]+b)+b') \\]
  * Loss: \\[ loss_t = -log P(w_t^*) \\]
* Pointer-Generator mechanism:
  * Soft-switch: \\[ p_{gen} = \sigma(w_{h^{\ast}}^T h_t^{\ast}+ w_s^T s_t+w_x^T x_t + b_{ptr}) \\] 
  * Probabilities becomes: \\[ P(w) = p_{gen}P_{vocab}(w)+(1-p_{gen})\sum_{i:w_i=w}{ai^t} \\]
* Coverage loss:
  * coverage vector: \\[ c^t = \sum_{t'=0}^{t-1}{a^{t'}} \\]
  * attn_scores: \\[ e_i^t = v^Ttanh(W_h h_i+W_s s_t+w_c c_i^t + b_{attn}) \\]
  * coverage loss: \\[ covloss_t = \sum_t min(a_i^t, c_i^t) \\]
* Final loss: \\[ loss_t = -log{P(w_t^*})+\lambda\sum_t{min(a_i^t, c_i^t)} \\]


# Paulus (2017) -- A Deep Reinforcement Model for Abstractive Summarization
*[[1705.04304]Romain Paulus, Caiming Xiong, Richard Socher, (2017), A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/abs/1705.04304)*

* Dataset: CNN/DM & NYT
* Implementation: no official. WIP.
* Results:   
  ![paulus_2017_results_cnndm](/img/paulus_2017_results_cnndm.png)   
  ![paulus_2017_results_nyt](/img/paulus_2017_results_nyt.png)   


### Model
* Intra-attention Model
   * attn scores: \\[ e_{ti} = {h_t^d}^T\\]
   * temporal scores: 
        \\[
            e^\prime_{ji} =
                \begin{cases}
                    exp(e_{ti}) & \text{if } t=1\\\\\\
                    \frac{exp(e_{ji})}{\sum_{j=1}^{t-1}{exp(e_{ji})}} & \text{otherwise}
                \end{cases}
        \\]
