---
layout: post
title: Abstractive Text Summarization State-of-the-Art
subtitle: A review of last models
---

# A. See (2017)
*[[1704.04368] Abigail See, Peter J. Liu, Christopher D. Manning, (2017) Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)

* Attention over source: 
  * $  e_i^t = v^T tanh(W_h h_i + W_s s_t + b_{attn}) $
  * $  a^t = softmax(e^t) $
  * $ h_t^* = \sum_i{a_i^th_i} $
  * $ P_{vocab}= softmax(V'(V[s_t, h_t^*]+b)+b') $
  * $ loss_t = -log P(w_t^*) $
* Pointer-Generator mechanism:
  * using soft-switch: $p_{gen} = \sigma(w_{h^*}^Th_t^*+w_s^Ts_t+w_x^Tx_t+b_{ptr})$
  * $P(w) = p_{gen}P_{vocab}(w)+(&-p_{gen})\sum_{i:w_i=w}ai^t$
* Coverage loss:
  * coverage vector: $ c^t = \sum_{t'=0}^{t-1}{a^{t'}} $ 
  * attn_scores: $ e_i^t = v^Ttanh(W_h h_i+W_s s_t+w_c c_i^t + b_{attn} $
  * coverage loss: $ covloss_t = \sum_t min(a_i^t, c_i^t) $
* Final loss: $loss_t = -log{P(w_t^*})+\lambda\sum_t{min(a_i^t, c_i^t)}$
