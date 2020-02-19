# -*- coding: utf-8 -*-
"""
Already Support: BERT
TODO: add mode: XLNET, ALBERT, RoBERTa
"""
import os
import collections
import pickle

import numpy as np
from absl import flags,logging,app
import tensorflow as tf

from posner.applications import bert
from posner.applications import bert_crf
from posner.optimizers import AdamWarmup
from posner.metrics import precision, recall,f1
from posner.utils import bert_tokenization as tokenization
from posner.datasets import chinese_daily_ner

FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
  "vocab_file",
  None,
  "The vocabulary file that the BERT model was trained on.")


flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")



## Other parameters
flags.DEFINE_string(
    "use_focal_loss", False,
    "")

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
  "task_name",
  None,
  "The name of the task to train.")


flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")


flags.DEFINE_bool(
  "do_lower_case",
  True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool(
  "do_train",
  False,
  "Whether to run training.")

flags.DEFINE_bool(
  "do_eval",
  False,
  "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict",
  False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_string(
    "predict_input_file", None,
    "The input file for prediction.")

flags.DEFINE_integer(
  "train_batch_size",
  32,
  "Total batch size for training.")

flags.DEFINE_integer(
  "eval_batch_size",
  8,
  "Total batch size for eval.")

flags.DEFINE_integer(
  "predict_batch_size",
  8,
  "Total batch size for predict.")

flags.DEFINE_float(
  "learning_rate",
  5e-5,
  "The initial learning rate for Adam.")

flags.DEFINE_integer(
  "num_train_epochs",
  3,
  "Total number of training epochs to perform.")

flags.DEFINE_integer(
    "decay_steps", 0,
  "")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer(
  "save_checkpoints_steps",
  1000,
  "How often to save the model checkpoint.")

flags.DEFINE_integer(
  "iterations_per_loop",
  1000,
  "How many steps to make in each estimator call.")

flags.DEFINE_bool(
  "use_tpu",
  False,
  "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string(
  "middle_output",
  "middle_data",
  "Dir was used to store middle data!")


flags.DEFINE_bool(
  "crf",
  False,
  "use crf")


# legacy
class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text = text
    self.label = label

# legacy
class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               mask,
               segment_ids,
               label_ids,
               is_real_example=True):
    self.input_ids = input_ids
    self.mask = mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.is_real_example = is_real_example

# legacy
class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_data(cls, input_file):
    """Read a BIO data!"""
    rf = open(input_file, 'r')
    lines = []
    words = []
    labels = []
    for line in rf:
      word = line.strip().split(' ')[0]
      label = line.strip().split(' ')[-1]
      if len(line.strip()) == 0 and words[-1] == '.':
        l = ' '.join([label for label in labels if len(label) > 0])
        w = ' '.join([word for word in words if len(word) > 0])
        lines.append((l, w))
        words = []
        labels = []
      words.append(word)
      labels.append(label)
    rf.close()
    return lines

# legacy
class NerProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_example(
      self._read_data(os.path.join(data_dir, "train.txt")), "train"
    )

  def get_dev_examples(self, data_dir):
    return self._create_example(
      self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
    )

  def get_test_examples(self, data_dir):
    return self._create_example(
      self._read_data(os.path.join(data_dir, "test.txt")), "test"
    )

  def get_labels(self):
    return ['B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O']

  def _create_example(self, lines, set_type): # set_type = {train dev test}
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      texts = tokenization.convert_to_unicode(line[1])
      labels = tokenization.convert_to_unicode(line[0])
      examples.append(InputExample(guid=guid, text=texts, label=labels))
    return examples

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, mode):
  """
  example:[Jim,Hen,##son,was,a,puppet,##eer]
  labels: [I-PER,I-PER,X,O,O,O,X]

  Args:
    ex_index: example num
    example: all labels
    label_list:
    max_seq_length:
    tokenizer: WordPiece tokenization
    mode:

  Returns:
    feature

  """
  label_map = {}
  # here start with zero this means that "[PAD]" is zero
  for (i, label) in enumerate(label_list):
    label_map[label] = i
  with open(FLAGS.middle_output + "/label2id.pkl", 'wb') as w:
    pickle.dump(label_map, w)
  textlist = example.text.split(' ')
  labellist = example.label.split(' ')
  tokens = []
  labels = []
  for i, (word, label) in enumerate(zip(textlist, labellist)):
    token = tokenizer.tokenize(word)
    tokens.extend(token)
    for i, _ in enumerate(token):
      if i == 0:
        labels.append(label)
      else:
        labels.append("X")
  # only Account for [CLS] with "- 1".
  if len(tokens) >= max_seq_length - 1:
    tokens = tokens[0:(max_seq_length - 1)]
    labels = labels[0:(max_seq_length - 1)]
  ntokens = []
  segment_ids = []
  label_ids = []
  ntokens.append("[CLS]")
  segment_ids.append(0)
  label_ids.append(label_map["[CLS]"])
  for i, token in enumerate(tokens):
    ntokens.append(token)
    segment_ids.append(0)
    label_ids.append(label_map[labels[i]])
  # after that we don't add "[SEP]" because we want a sentence don't have
  # stop tag, because i think its not very necessary.
  # or if add "[SEP]" the model even will cause problem, special the crf layer was used.
  input_ids = tokenizer.convert_tokens_to_ids(ntokens)
  mask = [1] * len(input_ids)
  # use zero to padding and you should
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    mask.append(0)
    segment_ids.append(0)
    label_ids.append(0)
    ntokens.append("[PAD]")
  assert len(input_ids) == max_seq_length
  assert len(mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(label_ids) == max_seq_length
  assert len(ntokens) == max_seq_length
  if ex_index < 3:
    logging.info("*** Example ***")
    logging.info("guid: %s" % (example.guid))
    logging.info("tokens: %s" % " ".join(
      [tokenization.printable_text(x) for x in tokens]))
    logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
    logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
  feature = InputFeatures(
    input_ids=input_ids,
    mask=mask,
    segment_ids=segment_ids,
    label_ids=label_ids,
  )
  # we need ntokens because if we do predict it can help us return to original token.
  return feature, ntokens, label_ids

def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  name_to_features = {
    "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
    "mask": tf.FixedLenFeature([seq_length], tf.int64),
    "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    "label_ids": tf.FixedLenFeature([seq_length], tf.int64),

  }

  def _decode_record(record, name_to_features):
    example = tf.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t
    return example

  def input_fn(params):
    batch_size = params["batch_size"]
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)
    d = d.apply(tf.data.experimental.map_and_batch(
      lambda record: _decode_record(record, name_to_features),
      batch_size=batch_size,
      drop_remainder=drop_remainder
    ))
    return d

  return input_fn

def filed_based_convert_examples_to_features(examples, label_list,
                                             max_seq_length, tokenizer,
                                             output_file, mode=None):
  writer = tf.python_io.TFRecordWriter(output_file)
  batch_tokens = []
  batch_labels = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 5000 == 0:
      logging.info("Writing example %d of %d" % (ex_index, len(examples)))
    feature, ntokens, label_ids = convert_single_example(ex_index, example,
                                                         label_list,
                                                         max_seq_length,
                                                         tokenizer, mode)
    batch_tokens.extend(ntokens)
    batch_labels.extend(label_ids)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["mask"] = create_int_feature(feature.mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature(feature.label_ids)
    tf_example = tf.train.Example(
      features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  # sentence token in each batch
  writer.close()
  return batch_tokens, batch_labels


def Writer(output_predict_file, result, batch_tokens, batch_labels, id2label):
  def _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i):
    token = batch_tokens[i]
    predict = id2label[prediction]
    true_l = id2label[batch_labels[i]]
    if token != "[PAD]" and token != "[CLS]" and true_l != "X":
      #
      if predict == "X" and not predict.startswith("##"):
        predict = "O"
      line = "{}\t{}\t{}\n".format(token, true_l, predict)
      wf.write(line)
  with open(output_predict_file, 'w') as wf:

    if FLAGS.crf:
      predictions = []
      for m, pred in enumerate(result):
        predictions.extend(pred)
      for i, prediction in enumerate(predictions):
        _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i)

    else:
      for i, prediction in enumerate(result):
        _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i)


def train_ner():
  tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  ## TODO: update for nother datasets.
  # processor = NerProcessor()
  # train_examples = processor.get_train_examples(FLAGS.data_dir)
  # label_list = processor.get_labels()
  # output_dims = len(label_list)

  (x_train, y_train), (x_test, y_test), (vocab, pos_tags) = \
    chinese_daily_ner.load_data(path=None, maxlen=FLAGS.max_seq_length, onehot=True, min_freq=2)
  output_dims = len(pos_tags)

  num_train_steps = int(
      len(x_train) * FLAGS.num_train_epochs / FLAGS.train_batch_size)

  if FLAGS.crf:
    model = bert_crf.load_trained_model_from_checkpoint(
      config_file=FLAGS.bert_config_file,
      checkpoint_file=FLAGS.init_checkpoint,
      crf_dims=output_dims,
      training=True,
      seq_len=FLAGS.max_seq_length,
    )

  else:
    model = bert.load_trained_model_from_checkpoint(
      config_file=FLAGS.bert_config_file,
      checkpoint_file=FLAGS.init_checkpoint,
      training=True,
      seq_len=FLAGS.max_seq_length,
    )

    bottle = tf.keras.layers.Dense(output_dims, activation='softmax', name='NER-output')
    inp = model.input
    out = bottle(model.layers[-9].output) # exlude MLM, NSP
    model = tf.keras.models.Model(inp, out)

    model.summary(line_length=150)

  logging.info("***** Running training *****")
  logging.info("  Num examples = %d", len(x_train))
  logging.info("  Batch size = %d", FLAGS.train_batch_size)
  logging.info("  Num steps = %d", num_train_steps)

  warmup_steps=int(num_train_steps*FLAGS.warmup_proportion)
  optimizer = AdamWarmup(decay_steps=FLAGS.decay_steps,warmup_steps=warmup_steps)

  if FLAGS.use_focal_loss:
    #TODO: test CategoricalFocalLoss
    from posner.losses.focal_loss import CategoricalFocalLoss
    focal_loss = CategoricalFocalLoss()
    model.compile(optimizer=optimizer,
                  loss=focal_loss,
                  metrics=[precision,recall,f1])
  else:
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=[precision,recall,f1])

  model.fit([x_train, np.zeros_like(x_train),np.ones_like(x_train)],
            y_train,
            epochs=FLAGS.num_train_epochs)
  return model


def eval_ner(model):
  (x_train, y_train), (x_test, y_test), (vocab, pos_tags) = \
    chinese_daily_ner.load_data(path=None, maxlen=FLAGS.max_seq_length, onehot=True, min_freq=2)
  logging.info("***** Running evaluation *****")
  logging.info("  Num examples = %d", len(x_test))
  logging.info("  Batch size = %d", FLAGS.eval_batch_size)
  _, precition, recall, f1 = model.evaluate(x=x_test, y=y_test)
  logging.info("***********************************************")
  logging.info("*********** Precision = %s*********************", str(precision))
  logging.info("************** Recall = %s*********************", str(recall))
  logging.info("****************** F1 = %s*********************", str(f1))
  logging.info("***********************************************")

  return model



def predict_ner(model):
  def _process_data(data, character, maxlen=None):
    if maxlen is None:
      maxlen = max(len(s) for s in data[0])

    word2idx = dict((w, i) for i, w in enumerate(character))

    x = []
    for s in data[0]:
      temp = [word2idx.get(c, 1) for c in s]
      x.append(temp)

    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen)
    return x


  (_,_),(_,_), (character, pos_tags) = \
  chinese_daily_ner.load_data()

  with open(FLAGS.predict_input_file, 'r') as f:
    text = f.readlines()

  logging.info("***** Running prediction*****")
  logging.info("  Num examples = %d", len(text))

  x = _process_data(text, character, maxlen=None)
  y = model.predict(x)
  y = np.argmax(y, axis=0)
  y = [pos_tags[i] for i in list(y)]



  with open(os.path.join(FLAGS.output_dir, "label_test.txt"), 'w+') as f:
    for char, ner in zip(list(x), y):
      f.write('{} {}\n'.format(char, ner))

def main(_):
  logging.set_verbosity(logging.INFO)

  if FLAGS.do_train:
    model = train_ner()

  if FLAGS.do_eval:
    model=eval_ner(model)

  if FLAGS.do_predict:
    model=predict_ner(model)


if __name__ == "__main__":
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("output_dir")
  app.run(main)
