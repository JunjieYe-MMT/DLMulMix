#!/bin/bash


python3 scripts/average_checkpoints.py \
			--inputs results/en-de/mmtimg \
			--num-epoch-checkpoints 11 \
			--output results/en-de/mmtimg/model.pt \
