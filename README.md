# DLMulMix
 Dual-level Interactive Multimodal-Mixup Encoder for Multi-modal Neural Machine Translation
## Requirements
ubuntu  
cuda==11.2  
python==3.7  
torch==1.8.1+cu111  
fairseq==0.9.0  

## Dataset
Text data we employ the dataset [Multi30K data set](http://www.statmt.org/wmt18/multimodal-task.html), then use [BPE](https://github.com/rsennrich/subword-nmt) to preprocess the raw data(dataset/data/task1/tok/). Image features are extracted through the pre-trained Resnet-101 and Faster-RCNN.  
The data-raw floder above is the data processed by BPE.
#### BPE (learn_joint_bpe_and_vocab.py and apply_bpe.py)
English, German, French use BPE participle separately.   
-s 10000  \
--vocabulary-threshold 1 \
## DLMulMix coder Quickstart
Step 1: bash data-preprocess.sh  
step 2: bash data-train.sh  
step 3: bash data-checkpoints.sh  
step 4: bash data-generate.sh  

Our model need to train 60 batches.  
The data-bin folder is the text data processed by bash data-preprocess.sh. Add the extracted image features here to start training the model.
![41 77](https://user-images.githubusercontent.com/90311581/138554952-face46fd-12e3-4cfb-ba3a-e9babd046777.jpg)

## Reproduce Existing Methods  
Doubly-ATT. [fairseq-Doubly-att.zip](https://github.com/DLMulMix/DLMulMix/files/7895802/fairseq-Doubly-att.zip)
