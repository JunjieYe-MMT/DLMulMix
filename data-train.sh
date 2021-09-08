#!/bin/bash

python train.py data-bin/en-de \
		--arch transformer_iwslt_de_en \
		--share-decoder-input-output-embed \
		--clip-norm 0 \
		--optimizer adam \
		--reset-optimizer \
		--lr 0.0011 \
		--source-lang en \
		--target-lang de \
		--max-tokens 1536 \
		--no-progress-bar \
		--log-interval 100 \
		--min-lr 1e-09 \
		--weight-decay 0.0001 \
		--criterion label_smoothed_cross_entropy \
		--label-smoothing 0.2 \
		--lr-scheduler inverse_sqrt \
		--max-update 47000 \
		--warmup-updates 4000 \
		--warmup-init-lr 1e-07 \
		--update-freq 4 \
		--adam-betas 0.9,0.98 \
		--keep-last-epochs 20 \
		--dropout 0.3 \
		--tensorboard-logdir results/en-de-bpe/bl_log1 \
		--log-format simple \
		--save-dir results/pre_mixup/mmtimg_0.2 \
		--eval-bleu \
		--patience 10 \
		--fp16 \

