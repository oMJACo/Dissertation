# coding: utf-8

import argparse
import os
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets
from music_dataset import MusicDataset
import model

parser = argparse.ArgumentParser(description='PyTorch Music Dataset Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--lr_decay', type=float, default=0.95,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=200,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

if not os.path.exists(args.save):
    os.makedirs(args.save)

open(args.save + '/summary.txt','w').close()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
print('Loading data...')
# set up fields
TEXT = data.Field(lower=False, batch_first=True, eos_token="<eos>")

# make splits for data
train_data = MusicDataset(path='data/',
                              exts=('train_src', 'train_tgt'),
                              fields=[('src',TEXT),('trg',TEXT)])

val_data = MusicDataset(path='data/',
                              exts=('valid_src', 'valid_tgt'),
                              fields=[('src',TEXT),('trg', TEXT)])

# build the vocabulary
TEXT.build_vocab(train_data)
vocab_length = len(TEXT.vocab)

# make iterator for splits
train_iter, valid_iter = data.BucketIterator.splits(
    (train_data, val_data), batch_sizes=(50,50), device=0, repeat=False)

###############################################################################
# Build the model
###############################################################################

print('Building the model...')
ntokens = vocab_length
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    val_batch_number = 0
    ntokens = vocab_length
    hidden = model.init_hidden(args.batch_size)

    for batch in valid_iter:

        val_batch_number += 1

        data = batch.src.transpose(0,1)
        targets = batch.trg.transpose(0,1)
        targets.contiguous()

        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets.view(-1)).data
        hidden = repackage_hidden(hidden)

    return total_loss[0] / len(data_source)

print('Starting Training')

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    batch_number = 0
    start_time = time.time()
    ntokens = vocab_length
    hidden = model.init_hidden(args.batch_size)

    for batch in train_iter:

        batch_number += 1

        data = batch.src.transpose(0,1)
        targets = batch.trg.transpose(0,1)
        targets.contiguous()

        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)

        loss = criterion(output.view(-1,ntokens), targets.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch_number % args.log_interval == 0 and batch_number > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch_number, len(train_data) // args.batch_size, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))

            with open(args.save + '/summary.txt', 'a') as summary_file:
                summary_file.write('loss {:5.2f}, ppl {:8.2f}\n'.format(cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))

        with open(args.save + '/summary.txt', 'a') as summary_file:

            summary_file.write('| End of epoch {:3d} | time: {:5.2f}s | valid_loss {:5.2f} | '
                                 'valid ppl {:8.2f}\n'.format(epoch,
                                                            (time.time() - epoch_start_time),
                                                            val_loss, math.exp(val_loss)))


        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save + '/model.pt', 'wb') as f:
                print('Saving best model at epoch ' + str(epoch) + '\n')
                torch.save(model, f)
            best_val_loss = val_loss
            with open(args.save + '/summary.txt', 'a') as summary_file:
                summary = ('Best val perplexity: ' + str(math.exp(best_val_loss)) + ' at epoch ' + str(epoch) + '\n')
                print(summary)
                summary_file.write(summary)

        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr *= args.lr_decay

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early with best val perplexity: ' + str(math.exp(best_val_loss)))

# Load the best saved model.
with open(args.save + '/model.pt', 'rb') as f:
    model = torch.load(f)


