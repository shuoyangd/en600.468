import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable
# from example_module import BiRNNLM
from model import BiRNNLM

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Starter code for JHU CS468 Machine Translation HW4.")
parser.add_argument("--data_file", required=True,
                    help="File for training set.")
parser.add_argument("--model_file", required=True,
                    help="Location to dump the models.")
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


def main(options):

  use_cuda = (len(options.gpuid) >= 1)
  if options.gpuid:
    cuda.set_device(options.gpuid[0])

  train, dev, test, vocab = torch.load(open(options.data_file, 'rb'), pickle_module=dill)

  batched_train, batched_train_mask, _ = utils.tensor.advanced_batchize(train, options.batch_size, vocab.stoi["<pad>"])
  batched_dev, batched_dev_mask, _ = utils.tensor.advanced_batchize(dev, options.batch_size, vocab.stoi["<pad>"])

  vocab_size = len(vocab)

  rnnlm = BiRNNLM(vocab_size)
  if use_cuda > 0:
    rnnlm.cuda()
  else:
    rnnlm.cpu()

  criterion = torch.nn.NLLLoss()
  optimizer = eval("torch.optim." + options.optimizer)(rnnlm.parameters(), options.learning_rate)

  # main training loop
  last_dev_avg_loss = float("inf")
  for epoch_i in range(options.epochs):
    logging.info("At {0}-th epoch.".format(epoch_i))
    # srange generates a lazy sequence of shuffled range
    for i, batch_i in enumerate(utils.rand.srange(len(batched_train))):
      train_batch = Variable(batched_train[batch_i])  # of size (seq_len, batch_size)
      train_mask = Variable(batched_train_mask[batch_i])
      if use_cuda:
        train_batch = train_batch.cuda()
        train_mask = train_mask.cuda()

      sys_out_batch = rnnlm(train_batch)  # (seq_len, batch_size, vocab_size) # TODO: substitute this with your module
      train_in_mask = train_mask.view(-1)
      train_in_mask = train_in_mask.unsqueeze(1).expand(len(train_in_mask), vocab_size)
      train_out_mask = train_mask.view(-1)
      sys_out_batch = sys_out_batch.view(-1, vocab_size)
      train_out_batch = train_batch.view(-1)
      sys_out_batch = sys_out_batch.masked_select(train_in_mask).view(-1, vocab_size)
      train_out_batch = train_out_batch.masked_select(train_out_mask)
      loss = criterion(sys_out_batch, train_out_batch)
      logging.debug("loss at batch {0}: {1}".format(i, loss.data[0]))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # validation -- this is a crude esitmation because there might be some paddings at the end
    dev_loss = 0.0
    for batch_i in range(len(batched_dev)):
      dev_batch = Variable(batched_dev[batch_i])
      dev_mask = Variable(batched_dev_mask[batch_i])
      if use_cuda:
        dev_batch = dev_batch.cuda()
        dev_mask = dev_mask.cuda()

      sys_out_batch = rnnlm(dev_batch)
      dev_in_mask = dev_mask.view(-1)
      dev_in_mask = dev_in_mask.unsqueeze(1).expand(len(dev_in_mask), vocab_size)
      dev_out_mask = dev_mask.view(-1)
      sys_out_batch = sys_out_batch.view(-1, vocab_size)
      dev_out_batch = dev_batch.view(-1)
      sys_out_batch = sys_out_batch.masked_select(dev_in_mask).view(-1, vocab_size)
      dev_out_batch = dev_out_batch.masked_select(dev_out_mask)
      loss = criterion(sys_out_batch, dev_out_batch)
      dev_loss += loss
    dev_avg_loss = dev_loss / len(batched_dev)
    logging.info("Average loss value per instance is {0} at the end of epoch {1}".format(dev_avg_loss.data[0], epoch_i))

    if (last_dev_avg_loss - dev_avg_loss).data[0] < options.estop:
      logging.info("Early stopping triggered with threshold {0} (previous dev loss: {1}, current: {2})".format(epoch_i, last_dev_avg_loss.data[0], dev_avg_loss.data[0]))
      break
    torch.save(rnnlm, open(options.model_file + ".nll_{0:.2f}.epoch_{1}".format(dev_avg_loss.data[0], epoch_i), 'wb'), pickle_module=dill)
    last_dev_avg_loss = dev_avg_loss


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
