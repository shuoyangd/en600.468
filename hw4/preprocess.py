import argparse
import codecs
import dill
import collections
import logging

import torch
import torchtext.vocab

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Starter code for JHU CS468 Machine Translation HW4.")
parser.add_argument("--train_file", required=True,
                    help="Training text file that needs to be preprocessed.")
parser.add_argument("--dev_file", required=True,
                    help="Dev text file that needs to be preprocessed.")
parser.add_argument("--test_file", required=True,
                    help="Test text file that needs to be preprocessed.")
parser.add_argument("--data_file", required=True,
                    help="Path to store the binarized data file.")
parser.add_argument("--min_count", default=5, type=int,
                    help="Count threshold for the token to not be considered as <unk>. (default=5)")
parser.add_argument("--charniak", action="store_true", default=False,
                    help="If the original text is already in charniak format " +
                    "(having <s> </s> at the beginning and end), enable this option to suppress padding.")
parser.add_argument("--vocab_size", default=None, type=int,
                    help="Maximum vocabulary size.")

UNK = "<unk>"
PAD = "<pad>"
BLK = "<blank>"


def main(options):
  # first pass: collect count
  token_cnt = collections.Counter()
  for line in codecs.open(options.train_file, 'r', 'utf8'):
    tokens = line.split()
    for token in tokens:
      token_cnt[token] += 1
    if not options.charniak:
      token_cnt["<s>"] += 1
      token_cnt["</s>"] += 1

  # filter min_count
  to_del = []
  token_cnt[UNK] = 0
  for key in token_cnt.keys():
    if token_cnt[key] < options.min_count:
      token_cnt[UNK] += token_cnt[key]
      to_del.append(key)

  for key in to_del:
    del token_cnt[key]

  vocab = torchtext.vocab.Vocab(token_cnt, max_size=options.vocab_size, specials=[PAD, BLK])

  # second pass: numberize
  train_data = []
  for line in codecs.open(options.train_file, 'r', 'utf8'):
    tokens = line.split()
    token_ids = []
    if not options.charniak:
      token_ids.append(vocab.stoi["<s>"])
    for token in tokens:
      token_ids.append(vocab.stoi[token])
    if not options.charniak:
      token_ids.append(vocab.stoi["</s>"])
    sent = torch.LongTensor(token_ids)
    train_data.append(sent)

  dev_data = []
  for line in codecs.open(options.dev_file, 'r', 'utf8'):
    tokens = line.split()
    token_ids = []
    if not options.charniak:
      token_ids.append(vocab.stoi["<s>"])
    for token in tokens:
      token_ids.append(vocab.stoi[token])
    if not options.charniak:
      token_ids.append(vocab.stoi["</s>"])
    sent = torch.LongTensor(token_ids)
    dev_data.append(sent)

  test_data = []
  for line in codecs.open(options.test_file, 'r', 'utf8'):
    tokens = line.split()
    token_ids = []
    if not options.charniak:
      token_ids.append(vocab.stoi["<s>"])
    for token in tokens:
      token_ids.append(vocab.stoi[token])
    if not options.charniak:
      token_ids.append(vocab.stoi["</s>"])
    sent = torch.LongTensor(token_ids)
    test_data.append(sent)

  torch.save((train_data, dev_data, test_data, vocab), open(options.data_file, 'wb'), pickle_module=dill)


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
