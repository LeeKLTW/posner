# POSNER
A toolkit implementing state-of-art models in Chinese word segmentation(CWS), part-of-speech (POS), and Name Entity Recognition(NER).

## Getting Started
### Prerequisites
1. Create a python virtual environment
`virtualenv venv`

2. source
`source venv/bin/activate`

3. install required python package
`pip install -r requirements.txt`

4. download pretrained word embeddings
download `https://github.com/ymcui/Chinese-BERT-wwm` 's BERT-wwm-ext, Chinese from
`https://drive.google.com/open?id=1buMLEjdtrXE2c4G1rpsNGWEx7lUQ0RHi`

5. unzip and place it at folder, your folder will be like this:
```
.
├── README.md
├── chinese_wwm_L-12_H-768_A-12
├── posner
├── requirements.txt
├── run_ner.py
├── run_ner.sh
├── tests
└── venv

```

p.s. Dataset is predefined in `posner/datasets/chinese_daily_ner.py`, you don't have to download manually.

### Running the NER training & evaluation

in cmd, input command below:
```
export BERT_BASE_DIR=./chinese_wwm_L-12_H-768_A-12                                              1 ↵

python run_ner.py \
  --do_train=true \
  --do_eval=true \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/chinese_ner

```
You will see the training starting, and show model architecture, and the result will be showed at the end of training.
```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
Input-Token (InputLayer)        [(None, 128)]        0
__________________________________________________________________________________________________
Input-Segment (InputLayer)      [(None, 128)]        0
__________________________________________________________________________________________________
Embedding-Token (TokenEmbedding [(None, 128, 768), ( 16226304    Input-Token[0][0]
__________________________________________________________________________________________________
Embedding-Segment (Embedding)   (None, 128, 768)     1536        Input-Segment[0][0]
__________________________________________________________________________________________________
Embedding-Token-Segment (Add)   (None, 128, 768)     0           Embedding-Token[0][0]
                                                                 Embedding-Segment[0][0]
__________________________________________________________________________________________________
Embedding-Position (PositionEmb (None, 128, 768)     98304       Embedding-Token-Segment[0][0]
__________________________________________________________________________________________________
Embedding-Dropout (Dropout)     (None, 128, 768)     0           Embedding-Position[0][0]
__________________________________________________________________________________________________
Embedding-Norm (LayerNormalizat (None, 128, 768)     1536        Embedding-Dropout[0][0]
__________________________________________________________________________________________________
Encoder-1-MultiHeadSelfAttentio (None, 128, 768)     2362368     Embedding-Norm[0][0]
__________________________________________________________________________________________________
Encoder-1-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-1-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-1-MultiHeadSelfAttentio (None, 128, 768)     0           Embedding-Norm[0][0]
                                                                 Encoder-1-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-1-MultiHeadSelfAttentio (None, 128, 768)     1536        Encoder-1-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-1-FeedForward (Position (None, 128, 768)     4722432     Encoder-1-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-1-FeedForward-Dropout ( (None, 128, 768)     0           Encoder-1-FeedForward[0][0]
__________________________________________________________________________________________________
Encoder-1-FeedForward-Add (Add) (None, 128, 768)     0           Encoder-1-MultiHeadSelfAttention-
                                                                 Encoder-1-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-1-FeedForward-Norm (Lay (None, 128, 768)     1536        Encoder-1-FeedForward-Add[0][0]
__________________________________________________________________________________________________
Encoder-2-MultiHeadSelfAttentio (None, 128, 768)     2362368     Encoder-1-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
Encoder-2-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-2-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-2-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-1-FeedForward-Norm[0][0]
                                                                 Encoder-2-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-2-MultiHeadSelfAttentio (None, 128, 768)     1536        Encoder-2-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-2-FeedForward (Position (None, 128, 768)     4722432     Encoder-2-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-2-FeedForward-Dropout ( (None, 128, 768)     0           Encoder-2-FeedForward[0][0]
__________________________________________________________________________________________________
Encoder-2-FeedForward-Add (Add) (None, 128, 768)     0           Encoder-2-MultiHeadSelfAttention-
                                                                 Encoder-2-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-2-FeedForward-Norm (Lay (None, 128, 768)     1536        Encoder-2-FeedForward-Add[0][0]
__________________________________________________________________________________________________
Encoder-3-MultiHeadSelfAttentio (None, 128, 768)     2362368     Encoder-2-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
Encoder-3-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-3-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-3-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-2-FeedForward-Norm[0][0]
                                                                 Encoder-3-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-3-MultiHeadSelfAttentio (None, 128, 768)     1536        Encoder-3-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-3-FeedForward (Position (None, 128, 768)     4722432     Encoder-3-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-3-FeedForward-Dropout ( (None, 128, 768)     0           Encoder-3-FeedForward[0][0]
__________________________________________________________________________________________________
Encoder-3-FeedForward-Add (Add) (None, 128, 768)     0           Encoder-3-MultiHeadSelfAttention-
                                                                 Encoder-3-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-3-FeedForward-Norm (Lay (None, 128, 768)     1536        Encoder-3-FeedForward-Add[0][0]
__________________________________________________________________________________________________
Encoder-4-MultiHeadSelfAttentio (None, 128, 768)     2362368     Encoder-3-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
Encoder-4-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-4-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-4-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-3-FeedForward-Norm[0][0]
                                                                 Encoder-4-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-4-MultiHeadSelfAttentio (None, 128, 768)     1536        Encoder-4-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-4-FeedForward (Position (None, 128, 768)     4722432     Encoder-4-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-4-FeedForward-Dropout ( (None, 128, 768)     0           Encoder-4-FeedForward[0][0]
__________________________________________________________________________________________________
Encoder-4-FeedForward-Add (Add) (None, 128, 768)     0           Encoder-4-MultiHeadSelfAttention-
                                                                 Encoder-4-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-4-FeedForward-Norm (Lay (None, 128, 768)     1536        Encoder-4-FeedForward-Add[0][0]
__________________________________________________________________________________________________
Encoder-5-MultiHeadSelfAttentio (None, 128, 768)     2362368     Encoder-4-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
Encoder-5-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-5-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-5-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-4-FeedForward-Norm[0][0]
                                                                 Encoder-5-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-5-MultiHeadSelfAttentio (None, 128, 768)     1536        Encoder-5-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-5-FeedForward (Position (None, 128, 768)     4722432     Encoder-5-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-5-FeedForward-Dropout ( (None, 128, 768)     0           Encoder-5-FeedForward[0][0]
__________________________________________________________________________________________________
Encoder-5-FeedForward-Add (Add) (None, 128, 768)     0           Encoder-5-MultiHeadSelfAttention-
                                                                 Encoder-5-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-5-FeedForward-Norm (Lay (None, 128, 768)     1536        Encoder-5-FeedForward-Add[0][0]
__________________________________________________________________________________________________
Encoder-6-MultiHeadSelfAttentio (None, 128, 768)     2362368     Encoder-5-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
Encoder-6-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-6-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-6-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-5-FeedForward-Norm[0][0]
                                                                 Encoder-6-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-6-MultiHeadSelfAttentio (None, 128, 768)     1536        Encoder-6-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-6-FeedForward (Position (None, 128, 768)     4722432     Encoder-6-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-6-FeedForward-Dropout ( (None, 128, 768)     0           Encoder-6-FeedForward[0][0]
__________________________________________________________________________________________________
Encoder-6-FeedForward-Add (Add) (None, 128, 768)     0           Encoder-6-MultiHeadSelfAttention-
                                                                 Encoder-6-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-6-FeedForward-Norm (Lay (None, 128, 768)     1536        Encoder-6-FeedForward-Add[0][0]
__________________________________________________________________________________________________
Encoder-7-MultiHeadSelfAttentio (None, 128, 768)     2362368     Encoder-6-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
Encoder-7-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-7-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-7-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-6-FeedForward-Norm[0][0]
                                                                 Encoder-7-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-7-MultiHeadSelfAttentio (None, 128, 768)     1536        Encoder-7-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-7-FeedForward (Position (None, 128, 768)     4722432     Encoder-7-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-7-FeedForward-Dropout ( (None, 128, 768)     0           Encoder-7-FeedForward[0][0]
__________________________________________________________________________________________________
Encoder-7-FeedForward-Add (Add) (None, 128, 768)     0           Encoder-7-MultiHeadSelfAttention-
                                                                 Encoder-7-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-7-FeedForward-Norm (Lay (None, 128, 768)     1536        Encoder-7-FeedForward-Add[0][0]
__________________________________________________________________________________________________
Encoder-8-MultiHeadSelfAttentio (None, 128, 768)     2362368     Encoder-7-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
Encoder-8-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-8-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-8-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-7-FeedForward-Norm[0][0]
                                                                 Encoder-8-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-8-MultiHeadSelfAttentio (None, 128, 768)     1536        Encoder-8-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-8-FeedForward (Position (None, 128, 768)     4722432     Encoder-8-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-8-FeedForward-Dropout ( (None, 128, 768)     0           Encoder-8-FeedForward[0][0]
__________________________________________________________________________________________________
Encoder-8-FeedForward-Add (Add) (None, 128, 768)     0           Encoder-8-MultiHeadSelfAttention-
                                                                 Encoder-8-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-8-FeedForward-Norm (Lay (None, 128, 768)     1536        Encoder-8-FeedForward-Add[0][0]
__________________________________________________________________________________________________
Encoder-9-MultiHeadSelfAttentio (None, 128, 768)     2362368     Encoder-8-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
Encoder-9-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-9-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-9-MultiHeadSelfAttentio (None, 128, 768)     0           Encoder-8-FeedForward-Norm[0][0]
                                                                 Encoder-9-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-9-MultiHeadSelfAttentio (None, 128, 768)     1536        Encoder-9-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-9-FeedForward (Position (None, 128, 768)     4722432     Encoder-9-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-9-FeedForward-Dropout ( (None, 128, 768)     0           Encoder-9-FeedForward[0][0]
__________________________________________________________________________________________________
Encoder-9-FeedForward-Add (Add) (None, 128, 768)     0           Encoder-9-MultiHeadSelfAttention-
                                                                 Encoder-9-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-9-FeedForward-Norm (Lay (None, 128, 768)     1536        Encoder-9-FeedForward-Add[0][0]
__________________________________________________________________________________________________
Encoder-10-MultiHeadSelfAttenti (None, 128, 768)     2362368     Encoder-9-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
Encoder-10-MultiHeadSelfAttenti (None, 128, 768)     0           Encoder-10-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-10-MultiHeadSelfAttenti (None, 128, 768)     0           Encoder-9-FeedForward-Norm[0][0]
                                                                 Encoder-10-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-10-MultiHeadSelfAttenti (None, 128, 768)     1536        Encoder-10-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-10-FeedForward (Positio (None, 128, 768)     4722432     Encoder-10-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-10-FeedForward-Dropout  (None, 128, 768)     0           Encoder-10-FeedForward[0][0]
__________________________________________________________________________________________________
Encoder-10-FeedForward-Add (Add (None, 128, 768)     0           Encoder-10-MultiHeadSelfAttention
                                                                 Encoder-10-FeedForward-Dropout[0]
__________________________________________________________________________________________________
Encoder-10-FeedForward-Norm (La (None, 128, 768)     1536        Encoder-10-FeedForward-Add[0][0]
__________________________________________________________________________________________________
Encoder-11-MultiHeadSelfAttenti (None, 128, 768)     2362368     Encoder-10-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
Encoder-11-MultiHeadSelfAttenti (None, 128, 768)     0           Encoder-11-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-11-MultiHeadSelfAttenti (None, 128, 768)     0           Encoder-10-FeedForward-Norm[0][0]
                                                                 Encoder-11-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-11-MultiHeadSelfAttenti (None, 128, 768)     1536        Encoder-11-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-11-FeedForward (Positio (None, 128, 768)     4722432     Encoder-11-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-11-FeedForward-Dropout  (None, 128, 768)     0           Encoder-11-FeedForward[0][0]
__________________________________________________________________________________________________
Encoder-11-FeedForward-Add (Add (None, 128, 768)     0           Encoder-11-MultiHeadSelfAttention
                                                                 Encoder-11-FeedForward-Dropout[0]
__________________________________________________________________________________________________
Encoder-11-FeedForward-Norm (La (None, 128, 768)     1536        Encoder-11-FeedForward-Add[0][0]
__________________________________________________________________________________________________
Encoder-12-MultiHeadSelfAttenti (None, 128, 768)     2362368     Encoder-11-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
Encoder-12-MultiHeadSelfAttenti (None, 128, 768)     0           Encoder-12-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-12-MultiHeadSelfAttenti (None, 128, 768)     0           Encoder-11-FeedForward-Norm[0][0]
                                                                 Encoder-12-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-12-MultiHeadSelfAttenti (None, 128, 768)     1536        Encoder-12-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-12-FeedForward (Positio (None, 128, 768)     4722432     Encoder-12-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-12-FeedForward-Dropout  (None, 128, 768)     0           Encoder-12-FeedForward[0][0]
__________________________________________________________________________________________________
Encoder-12-FeedForward-Add (Add (None, 128, 768)     0           Encoder-12-MultiHeadSelfAttention
                                                                 Encoder-12-FeedForward-Dropout[0]
__________________________________________________________________________________________________
Encoder-12-FeedForward-Norm (La (None, 128, 768)     1536        Encoder-12-FeedForward-Add[0][0]
__________________________________________________________________________________________________
Input-Masked (InputLayer)       [(None, 128)]        0
__________________________________________________________________________________________________
NER-output (Dense)              (None, 128, 7)       5383        Encoder-12-FeedForward-Norm[0][0]
==================================================================================================
Total params: 101,387,527
Trainable params: 101,387,527
Non-trainable params: 0
__________________________________________________________________________________________________
I0214 12:41:21.480098 4651118016 run_ner.py:496] ***** Running training *****
I0214 12:41:21.480504 4651118016 run_ner.py:497]   Num examples = 20864
I0214 12:41:21.480728 4651118016 run_ner.py:498]   Batch size = 32
I0214 12:41:21.480917 4651118016 run_ner.py:499]   Num steps = 1956
Train on 20864 samples
Epoch 1/3
   64/20864 [..............................] - ETA: 29:21:53 - loss: 1.2914 - precision: 0.0036 - recall: 0.0020 - f1: 0.0025

...

```


## Implementation detail
### pretrained
Instead the original([BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)) pretrained model, 
whole word masking pretrained model(
[Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/pdf/1906.08101.pdf)) is recommended.
Because Whole Word Masking (WWM) mitigates the drawbacks of masking partial 
WordPiece tokens in pre-training BERT. For character based, it is easy to guess 
the character if another character in the same word is masked.

| 说明 | 样例 |
| :------- | :--------- |
| 原始文本 | 使用语言模型来预测下一个词的probability。 |
| 分词文本 | 使用 语言 模型 来 预测 下 一个 词 的 probability 。 |
| 原始Mask输入 | 使 用 语 言 [MASK] 型 来 [MASK] 测 下 一 个 词 的 pro [MASK] ##lity 。 |
| 全词Mask输入 | 使 用 语 言 [MASK] [MASK] 来 [MASK] [MASK] 下 一 个 词 的 [MASK] [MASK] [MASK] 。 |


### CRF
Conditional random fields (CRFs) is a statistical modeling method 
often applied structured prediction.Whereas a classifier predicts a label for a 
single sample without considering "neighboring" samples, a CRF can take context 
into account.  

Linear chain CRFs, which implement sequential dependencies in the 
predictions, is popular in NLP sequence labeling.

Though pure NN sequence model is capable of NER labeling, combined with CRF will 
make it more efficient.  For example in POS,

```
input: "学习出一个模型，然后再预测出一条指定"
expected output: 学/B 习/E 出/S 一/B 个/E 模/B 型/E ，/S 然/B 后/E 再/E 预/B 测/E ……
NN sequence model: 学/B 习/E 出/S 一/B 个/B 模/B 型/E ，/S 然/B 后/B 再/E 预/B 测/E ……
```

The B should not come after B(begin of word), and this can be eliminate in CRF.
 
 
### Activation function
Origin Transformer use 'relu', BERT use 'gelu'. The GELU nonlinearity weights 
inputs by their magnitude, rather than gates inputs by their sign as in ReLUs.
Research shows it outperform 'relu' & 'elu'.([Hendrycks and Gimpel, 2016](https://arxiv.org/pdf/1606.08415.pdf))


### Loss
Take Cinese Daily News NER dataset for example. There labels are 'B-LOC', 
'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O', however, most of characters 
are labeled as 'O', these caused unbalanced classification problem. This repo
use focal loss instead of cross entropy for labeling.
([Tsung-Yi et al,2018](https://arxiv.org/pdf/1708.02002.pdf))

### Optimizer & Learning rate scheduling
Use Adam with warmup.  Cyclical learning rate scheduling is also provided. 

### Metrics
F1 score from C.Manning.

### Dataset
Default dataset is Chinese Daily News NER. Data loader is provided in 
`posner.datasets.chinese_daily_ner`
The data looks like this:
```
相 O
比 O
之 O
下 O
， O
青 B-ORG
岛 I-ORG
海 I-ORG
牛 I-ORG
队 I-ORG
和 O
广 B-ORG
州 I-ORG
松 I-ORG
日 I-ORG
队 I-ORG
的 O
雨 O
中 O
之 O
战 O
虽 O
然 O
也 O
是 O
0 O
∶ O
0 O
```


### Inference time (future work)
Inference time is slow for NER. Looking forward to implementing 
[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/pdf/1909.11942).



## Acknowledgments
Reference:
### Code
* [google-research/bert](https://github.com/google-research/bert)
* [tensorflow/models](https://github.com/tensorflow/models)
* [tensorflow/addons](https://github.com/tensorflow/addons)
* [keras-team/keras-contrib](https://github.com/keras-team/keras-contrib)
* [CyberZHG/keras-bert](https://github.com/CyberZHG/keras-bert)
* [zjy-ucas/ChineseNER/](https://github.com/zjy-ucas/ChineseNER/)
* [BrikerMan/Kashgari](https://github.com/BrikerMan/Kashgari)
* [huggingface/transformers](https://github.com/huggingface/transformers)
* [fxsjy/jieba](https://github.com/fxsjy/jieba)
* [ckiplab/ckiptagger](https://github.com/ckiplab/ckiptagger)
* [ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)
* [hankcs/HanLP](https://github.com/hankcs/HanLP)


### Books & Paper & Website
* Daphne Koller & Nir Friedman. 2009, Probabilistic Graphical Models -- Principles and Techniques
* CD Manning, 2008, Introduction to Information Retrieval
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
* [Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/pdf/1906.08101.pdf)
* [Bridging nonlinearities and stochastic regularizers with gaussian error linear units](https://arxiv.org/pdf/1606.08415.pdf)
* [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)
* [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
* [State-of-the-art Chinese Word Segmentation with Bi-LSTMs](https://aclweb.org/anthology/D18-1529)
* [Subword Encoding in Lattice LSTM for Chinese Word Segmentation](https://arxiv.org/pdf/1810.12594.pdf)
* [Neural Word Segmentation with Rich Pretraining](http://aclweb.org/anthology/P17-1078)
* [Word-Context Character Embeddings for Chinese Word Segmentation](https://www.aclweb.org/anthology/D17-1079)
* [Adversarial Multi-Criteria Learning for Chinese Word Segmentation](http://aclweb.org/anthology/P17-1110)
* [Exploring Segment Representations for Neural Segmentation Models](https://www.ijcai.org/Proceedings/16/Papers/409.pdf)
* [Long Short-Term Memory Neural Networks for Chinese Word Segmentation](http://www.aclweb.org/anthology/D15-1141)
* [Neural Joint Model for Transition-based Chinese Syntactic Analysis](http://www.aclweb.org/anthology/P17-1111)
* [Fast and Accurate Neural Word Segmentation for Chinese](http://aclweb.org/anthology/P17-2096)
* [Neural Word Segmentation Learning for Chinese](http://www.aclweb.org/anthology/P16-1039)
* [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/pdf/1909.11942)
* [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
* [nlpprogress](https://nlpprogress.com/)


## TODO list
- [ ] API like [fxsjy/jieba](https://github.com/fxsjy/jieba)
- [ ] word weight like [fxsjy/jieba](https://github.com/fxsjy/jieba)
- [ ] XLNet
- [ ] Albert
- [ ] RoBERTa
- [ ] Inference time optimize


