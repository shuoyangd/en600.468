import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable
from example_module import RNNLM
# from model import RNNLM

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
parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=SGD)")
parser.add_argument("--learning_rate", "-lr", default=0.1, type=float,
                    help="Learning rate of the optimization. (default=0.1)")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="Momentum when performing SGD. (default=0.9)")
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")
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

  use_cuda = (len(options.gpuid) >= 1)
  if options.gpuid:
    cuda.set_device(options.gpuid[0])

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
  if use_cuda > 0:
    rnnlm.cuda()
  else:
    rnnlm.cpu()

  criterion = torch.nn.NLLLoss()
  optimizer = eval("torch.optim." + options.optimizer)(rnnlm.parameters(), options.learning_rate)

  # main training loop
  last_dev_loss = float("inf")
  for epoch_i in range(options.epochs):
    logging.info("At {0}-th epoch.".format(epoch_i))
    # srange generates a lazy sequence of shuffled range
    for i, batch_i in enumerate(utils.rand.srange(len(batched_train_in))):
      train_in_batch = Variable(batched_train_in[batch_i])  # of size (seq_len, batch_size)
      train_out_batch = Variable(batched_train_out[batch_i])  # of size (seq_len, batch_size)
      if use_cuda:
        train_in_batch = train_in_batch.cuda()
        train_out_batch = train_out_batch.cuda()

      sys_out_batch = rnnlm(train_in_batch)  # (seq_len, batch_size, vocab_size) # TODO: substitute this with your module
      sys_out_batch = sys_out_batch.view(-1, vocab_size)
      train_out_batch = train_out_batch.view(-1)
      loss = criterion(sys_out_batch, train_out_batch)
      logging.debug("loss at batch {0}: {1}".format(i, loss.data[0]))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # validation -- this is a crude esitmation because there might be some paddings at the end
    dev_loss = 0.0
    for batch_i in range(len(batched_dev_in)):
      dev_in_batch = Variable(batched_dev_in[batch_i])
      dev_out_batch = Variable(batched_dev_out[batch_i])
      if use_cuda:
        dev_in_batch = dev_in_batch.cuda()
        dev_out_batch = dev_out_batch.cuda()

      sys_out_batch = rnnlm(dev_in_batch)
      sys_out_batch = sys_out_batch.view(-1, vocab_size)
      dev_out_batch = dev_out_batch.view(-1)
      loss = criterion(sys_out_batch, dev_out_batch)
      dev_loss += loss
    dev_avg_loss = dev_loss / len(batched_dev_in)
    logging.info("Average loss value per instance is {0} at the end of epoch {1}".format(dev_avg_loss.data[0], epoch_i))

    if (last_dev_loss - dev_loss).data[0] < options.estop:
      logging.info("Early stopping triggered with threshold {0} (previous dev loss: {1}, current: {2})".format(epoch_i, last_dev_loss, dev_loss))
      break
    last_dev_loss = dev_loss


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
