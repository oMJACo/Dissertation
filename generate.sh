#!/bin/bash

source activate pytorch35

CHECKPOINT=models/lstm_3_layers_lr_17_decay_95_rnntype_LSTM_dropout_0.2_hidden_512_emb_512/model.pt
WORDS=110
OUTFILE='generated_4_bar.txt'

python generate.py \
	--checkpoint ${CHECKPOINT} \
	--words ${WORDS} \
	--outf ${OUTFILE}

