---
layout: post
title: Implementing "A Deep Reinforcement Model for Abstractive Summarization" with OpenNMT-py
subtitle: Objectives, challenges and todo
---

# Model overview
The paper: <https://arxiv.org/pdf/1705.04304.pdf>
* Encoder 
    * Shared Embeddings (dim=100), vocab src=150k, tgt=50k
    * Bi-directionnal LSTM (dim=2x200)
* Decoder
    * LSTM (dim=400)
    * Encoder Temporal Attention (between $ h^d_t \text{ and } h_e^t $), producing attention scores ($a^e_t$) and context vector ($c^e_t$)
    * Decoder Attention (between $h^d_t$ and $h^t_{t-1}$ for $t>0$), producing attention scores ($a^d_t$) and     context vector ($c^e_t$)
* Pointer-Generator
    * Soft-switch $p(u_t=1)$ deciding to either generate ($u_t=0$) or copy ($u_t=1$)
    * Soft-switch value and generator scores are computed using decoder output ($h^d_t$), encoder context ($c^e_t$) and decoder context ($c^d_t$)
    * Pointer scores (what source token to copy) are intra-encoder attention scores ($a^e_t$)

* Tricks
    * To reduce exposure bias, we use predicted tokens and next step input for decoder (instead of ground truth) with a 25% probability. This force us to run prediction at decoding time, which isn't the case of other Open-NMT-models
    * Encoder and decoder embeddings shares their weights, but with different vocabulary size. Thus, the decoder embedding is part of the encoder embedding.
    * The output projection used in pointer generator (sect. (2.3), eq. (9)) shares weight with the target embedding matrix as described in sect. (2.4) eq. (13).

# Implementation
I am working on a public fork of OpenNMT-py, more precisely on the `drmas` branch: <https://github.com/pltrdy/OpenNMT-py/tree/drmas>.

Most of the implementation is in `onmt.Reinforced` for simplicity, to edit as much shared code as possible. We use a special flag `-reinforced` at training time to use our model. I try to avoid hardcoded variable as much as possible for obvious reason. Still, because it's still a very experimental code, one may find some.

## Embeddings
The paper states that encoder and decoder embedding must be shared, but, in the same time, `src_vocab > tgt_vocab`. This kind of embeddings does not exists so we implemented it as `onmt.modules.PartialEmbedding`.

The implementation is quite simple: the partial embedding `weight` parameter returns a slice of the `full_embedding` (i.e. the encoder emb). There is a kind of trick that shifts the embedding matrix. In fact, OpenNMT uses special tokens:
* **padding**: (token=`<pad>` id=1)
* **unknown**: (token=`<unk>` id=0)
* **sequence begining**: (token=`<s>` id=2)
* **sequence end**: (token=`<\s>` id=3)

The thing is that there is `<bos>` and `<eos>` only in the decoder. That's why PartialEmbedding has some extra parameters that are inserted in the weight matrix. The rest

## Encoder
There is nothing special in the encoder. We used `onmt.Encoder` for our bi-directional LSTM.

## Decoder
The decoder from DRMAS has different properties, thus we implemented our own class `onmt.Reinforced.ReinforcedDecoder`. 

First, it uses two kind of intra-attention. Second, it produces tokens at each decoding step (prediction) in order to be able to feed it as next-step input (with 0.25 probability) in order to reduce exposure bias as explained in sect. 6.1. So we run the generator and loss computation directly in the decoding loop. It's different than how other models works in OpenNMT-py. Generator/Loss computation is usually done after batch decoding, in the `onmt.Trainer`. For that reason, we use our own `onmt.Reinforced.RTrainer`. 

## Attention
The attention mechanism is defined in the `IntraAttention` class that is used both both intra-temporal and intra-decoder attention. There is two differences:
* **attention scores**: the `temporal (default=False)` set whether the mechanism will use temporal scores $e^{\prime}_{ti}$ as defined in sect. (2.1) eq. (3) or regular scores.
* **input**: in the forward pass, we use an $h_t$ arg, that correspond either to the encoder outputs $h^e_i$ or the decoder previous outputs $h^d_{t^{\prime}}$


It uses a `Linear(dim, dim)` and a softmax over scores. In case of temporal attention, the scores are softmax over previous timesteps.

## Pointer-Generator 
The pointer-generator part must produce: 
* **generator scores** $p(y_t \mid u_t=0)$ i.e. probabilities to generate tokens (of the target vocabulary); 
* **copy/pointer scores** $p(y_t \mid u_t=1)$ i.e. probabilities to copy source tokens; 
* **switch**: $p(u_t)$ weighting copy/generator scores at each step

The final distribution is `switch * copy + (1-switch) * generate` i.e. $p(y_t) = p(u_t=1) p(y_t \mid u_t=1) + p(u_t=0) p(y_t \mid u_t=0)$

The matrix used for the generator projection ($W_{out}$, eq. (9)) is itself a projection of the target embedding matrix: $W_{out} = tanh(W_{emb}W_{proj})$ (eq. (8)). Therefore, the $W_{out}$ matrix must be dynamically updated after each backward pass (which updates the embedding). The pointer-generator class therefore have a `W_out` **function** that returns the matrix, and update it (if `force=True`). We force update in `ReinforcedDecoder.forward()`. 

## LossCompute
We used a LossCompute function as it is the case for other OpenNMT models. The idea is to first get scores with a `generator`, then the loss using the `criterion` and finally return the loss and some stats. Because we use a copy mechanisms, we must apply what OpenNMT call `collapse_copy_scores` in order to sums copy scores with generation scores for same tokens. The thing is: a source token can be in the target vocabulary. If it's the case, then this token's score is the sum of its gen score and copy score. This is done using `Dataset.collapse_copy_scores` but it has been shown to be quite inefficient in our case, because we run it every decoding steps. It may be a good thing to get a faster version. In our case, we can use the fact that both vocabularies are shared, so token index are the same for token < target_vocab_size. 


We also adjust targets in order to take into account good out-of-vocabularies copies. Batches provides us an `alignment` vector that are targets tokens expressed in source vocabulary. Thanks to this, we can find cases where the target token was OOV (with respect to target vocab), thus `<unk>`, but the *true target* can be found by copy. The same logic is applied in the loss calculation to find which source *must* be copied.

## Trainer
Because our decoder runs generator/loss computation, it directly returns loss and stats, thus, our trainer looks simpler. Other differences are implementation details like args/returns order etc. 

## Translation
Some changes has been done to `onmt.Translator`. Usually just to fit with the previously mentioned implementation details.

## Debugging / Working with the code
In order to find out what was happening I used different tricks:
* I wrote a simple profiler in order to see what function was taking most of the time (it showed that collapse_copy_score was really time consuming, making my process ~25% GPU load)
* I'm used to print lot of information at decoding time like stacking input/target/prediction and attentions scores for 1 sequence at the end of decoding. It may shows if input/targets are properly aligned as well as if the copy mechanism is properly working or not.
* I did some attention matrix plot, which usually aren't really informative i.e. I'm just sure that something is broken without much clues. 

## Potential Issues
* Partial Embedding shifting is broken: I had issues with my custom shifting, input and target wasn't properly aligned. Checking input/target/prediction on simple case like copy was really helpfull here. I guess this is now ok, but I prefer sharing what issues I had to give ideas.
* Custom collapse copy score is broken: I tried some faster score collapse cpde, that was in fact faster, but incorrect, leading to really bad learning. I wrote it multiple time without much improvment. It's not as easy as it seems, or maybe I just don't manage to figure out the proper way to do it. 
* Attention Mechanism are incorrect: I had some issues on softmax computations. You may know that softmax is numerically unstable so doing it yourself (i.e. as in the paper: first exp, then divide) may lead to `NaN` loss. I think my current version that only uses builtin `nn.Softmax` is correct, but still, I can't overassume. 


## To try
* copy tasks: tasks that only involves copy. Nothing else must be learn. e.g. generate random sequence as input, and the same (or reversed) as target. The model might get a really high accuracy if it manage to copy. Try both with tokens in tgt vocab (copy scores collapsed) and tokens out of the tgt vocab (pure copy).
* Sort tasks: It just shows that the model can learn logic about tokens. It may be especially informative when using PartialEmbedding.
* I tried to run it on Gigaword, that must be easier/faster than CNN/DM. I ran into issues during preprocessing with `-dynamic_dict` because `preprocess.py` take way too much RAM. This is a known `torchtext` issue.

## Scripting

The CNN/DM dataset as in the link provided in the script contains some `</s>` that we want to remove.
I wrote a super simple `remove_eos.py` script for this:
```
#!/usr/bin/env python3
import sys 

for line in sys.stdin:
  if line == "": 
    continue
  if not line:
    break

  line = " ".join(line.replace('</s>', '.').split())

  print(line)
```

And a `run.sh` script that I usually copy to run different experiments (and set different `root|`) variable. Note that `./run.sh preprocess` puts the data into `$root` directory, making it possible to run experiments on different preprocessed data. Use `./run.sh train`, `./run.sh translate`, well I guess you got the point.
```
set -e

# first download data
# https://drive.google.com/uc?id=0B6N7tANPyVeBSGNkOWhTdGt6YXM&export=download
# and extract it to CNNDM_PATH (its an env variable on my system)
export cnndm_path="$SUMMARY_STORIES_DATA"
export data="./cnndm"
export gpu="0"
export root="experiment_0"

prepare(){
  echo "Retrieving CNN/DM data from $cnndm_path to $data"
  mkdir $data
  cp $cnndm_path/*.src.txt $data
  cp $cnndm_path/*.tgt.txt $data

  echo "Removing eos in place"
  for f in $(ls $data); do
    path="$data/$f"
    echo "processing $path..."
    cat $path | ./remove_eos.py > "${path}.tmp"
    mv "${path}.tmp" $path
  done

  echo "Done"
}

preprocess(){
  mkdir -p $root
  python preprocess.py \
      -train_src $data/train.src.txt \
      -train_tgt $data/train.tgt.txt \
      -valid_src $data/valid.src.txt \
      -valid_tgt $data/valid.tgt.txt \
      -src_seq_length 400 \
      -tgt_seq_length 100 \
      -save_data $root/data \
      -src_vocab_size 150000 \
      -tgt_vocab_size 50000 \
      -dynamic_dict
}

train(){
  # I usually cut training way before 100 epoch for sure
  # I just want it to run virtually for ever by default
  python train.py -data $root/data \
        -save_model $root/model \
        -batch_size 32 \
        -layers 2 \
        -brnn \
        -reinforced \
        -rnn_size 200 \
        -word_vec_size 100 \
        -rnn_type "LSTM" \
        -optim "adam" \
        -learning_rate 0.001 \
        -seed 10000 \
        -epochs 100 \
        -gpu "$gpu"
}

translate(){
  # get the last model, by date
  best_model=$(ls -lsrt $root/model*.pt | tail -n 1 | awk '{print $NF}')
  echo "Loading: $best_model"
  # do not set batch_size to anything but 1, it's broken
  python translate.py -model "$best_model" \
                      -src $data/test.src.txt \
                      -gpu "$gpu" \
                      -batch_size 1 \
                      -verbose \
                      -beam_size 5

  mv pred.txt "$root/pred.txt"
  echo "Translated to $root/pred.txt"

}

# Argument required
if [ -z "$1" ]; then
  echo "No action selected"
  echo "Usage: ./run.sh [prepare] [preprocess] [train] [translate]"
fi

# Read action and run it
for action in "$@"; do
  printf "\n****\nRunning command '$action'\n\n"
  eval "$action"
done

echo "Done" 
```


