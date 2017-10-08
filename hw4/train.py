import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch.autograd import Variable
# from example_module import RNNLM
from model import RNNLM

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Starter code for JHU CS468 Machine Translation HW4.")
parser.add_argument("--data_file", required=True,
                    help="File for training set.")
parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--optimizer", default="Adam",
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--learning_rate", "-lr", default=5e-2, type=float,
                    help="Learning rate of the optimization. (default=5e-2)")
# feel free to add more arguments as you need

def get_lm_input(data):
  input_data = []
  for sent in data:
    input_data.append(sent[:-1])
  return input_data


def get_lm_output(data):
  output_data = []
  for sent in data:
    output_data.append(sent[1:])
  return output_data


def main(options):
  train, dev, vocab = torch.load(open(options.data_file, 'rb'), pickle_module=dill)
  train_in = get_lm_input(train)
  train_out = get_lm_output(train)
  dev_in = get_lm_input(dev)
  dev_out = get_lm_output(dev)

  batched_train_in, _ = utils.tensor.advanced_batchize(train_in, options.batch_size, vocab.stoi["<pad>"])
  batched_train_out, _ = utils.tensor.advanced_batchize(train_out, options.batch_size, vocab.stoi["<pad>"])
  batched_dev_in, _ = utils.tensor.advanced_batchize(dev_in, options.batch_size, vocab.stoi["<pad>"])
  batched_dev_out, _ = utils.tensor.advanced_batchize(dev_out, options.batch_size, vocab.stoi["<pad>"])

  vocab_size = len(vocab)

  rnnlm = RNNLM(vocab_size)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = eval("torch.optim." + options.optimizer)(rnnlm.parameters(), options.learning_rate)

  # main training loop
  for epoch_i in range(options.epochs):
    logging.info("At {0}-th epoch.".format(epoch_i))
    # srange generates a lazy sequence of shuffled range
    for i, batch_i in enumerate(utils.rand.srange(len(batched_train_in))):
      train_in_batch = Variable(batched_train_in[batch_i])  # of size (seq_len, batch_size)
      train_out_batch = Variable(batched_train_out[batch_i])  # of size (seq_len, batch_size)
      """
      complete the training loop here with the module you implemented
      """
      sys_out_batch = rnnlm(train_in_batch)  # (seq_len, batch_size, vocab_size)
      sys_out_batch = sys_out_batch.view(-1, vocab_size)
      train_out_batch = train_out_batch.view(-1)
      loss = criterion(sys_out_batch, train_out_batch)
      logging.debug("loss at batch {0}: {1}".format(i, loss.data[0]))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
