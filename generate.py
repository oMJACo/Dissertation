import argparse
import torch
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets
import random


parser = argparse.ArgumentParser(description='Music Dataset Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/valid_src',
                    help='location of the data corpus')
parser.add_argument('--vocab', type=str, default='models/vocab.pt',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='models/lstm_3_layers_lr_18_decay_95_rrntype_LSTM/model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated_from_measure_seed.txt',
                    help='output file for generated text using measure seed')
parser.add_argument('--words', type=int, default='154',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=7,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--nBars', type=int, default=4,
                    help='Number of bars to use as seed')
args = parser.parse_args()

TEXT = data.Field(lower=False, batch_first=True, eos_token="<eos>")

train_data=datasets.MusicDataset(path='/home/mcowan/Project/folk-rnn/data/',
                                 exts=('train_src', 'train_tgt'),
                                 fields=[('src',TEXT),('trg',TEXT)])

val_data=datasets.MusicDataset(path='/home/mcowan/Project/folk-rnn/data/',
                               exts=('valid_src','valid_tgt'),
                               fields=[('src',TEXT),('trg', TEXT)])
import os.path
import pickle

vocab = None

if not os.path.isfile(args.vocab):
    print('Saving vocabulary...')
    TEXT.build_vocab(train_data)
    vocab = TEXT.vocab
    with open(args.vocab, 'wb') as f:
        pickle.dump(vocab, f)
else:
    print('Loading vocabulary...')
    vocab = pickle.load(open(args.vocab, 'rb'))

vocab_length = len(vocab)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

print('Loading model...')
with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    print('running on GPU')
    model.cuda()
else:
    print('running on CPU')
    model.cpu()

def firstNBars(n, tune):
    bar_count = 0
    firstNbars = []
    tune = tune.split()

    for token in tune:
        if token != '|' and token !=  '|:' and token != ':|':
            firstNbars.append(token)
        else:
            bar_count +=1
            if bar_count == n+1
                firstNbars.append(token)
                break
            else:
                firstNbars.append(token)
                continue
    return firstNbars

gold_seeds = []
gold_full = []

with open(args.data, 'r') as gold:
    for line in gold:
        gold_full.append(line[:-1])
        gold_seeds += [firstNBars(args.nBars, line[:-1])]


'''
with open('temp_val', 'r') as valFile:
    i = 0
    open('valid_no_seed_3_bar', 'w').close()
    for tune in valFile:
        print(i)
        tune = tune.split()
        seed_length = len(gold_seeds[i])
        tune_wo_seed = tune[seed_length:]
        i += 1
        with open('valid_no_seed_3_bar', 'a') as val3Bar:
            output_tune = ''
            for token in tune_wo_seed:
                output_tune += token + ' '
            val3Bar.write(output_tune + '\n')
'''#Code to remove seeds from files for BLEU Score tests

gold_seed_idxs = []

for tune in gold_seeds:
    tune_seed_idxs = []
    for token in tune:
        idx = vocab.stoi[token]
        tune_seed_idxs += [idx]
    gold_seed_idxs.append(tune_seed_idxs)

def genTune(tune_seed, file_ID, nWords):

    file_ID = str(file_ID)
    args.seed = random.choice(range(2000))

    hidden = model.init_hidden(1)
    output_tune = ''
    input = Variable(torch.LongTensor(1,1))
    input.data.fill_(tune_seed[0])
    word = vocab.itos[tune_seed[0]]

    output_tune += word

    if args.cuda:
        input.data = input.data.cuda()

    with open('GEN_REPORT_FROM' + file_ID, 'w') as outf:
        for i in range(nWords):
            if i < len(tune_seed) - 1:
                output, hidden = model(input, hidden)
                word_idxs = tune_seed[i+1]
            else:
                output, hidden = model(input,hidden)
                word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
                word_idxs = torch.multinomial(word_weights, 1)[0]

            input.data.fill_(word_idxs)
            word = vocab.itos[word_idxs]

            if word == '<eos>':
                break
            else:
                output_tune += word

        '''
        output_tune_list = output_tune.split()
        seed_length = len(tune_seed)
        output_wo_seed = output_tune_list[seed_length:]

        output_tune = ''

        for token in output_wo_seed:
            output_tune += token + ' '
        '''#Code for removing the seeds after generation

        outf.write(output_tune + '\n')
        return output_tune


for i in range(8):
    j = random.choice(range(2000))
    print(j)
    print(genTune(gold_seed_idxs[29],'1BAR',len(gold_full[29])))
#Generating samples from a partciular tune in the validation set


'''
print('Starting generation of tunes...')

tune_count = 0
j=0

for i in range(10):
    if os.path.isfile('GEN4BAR' + str(i)):
        open('GEN4BAR' + str(i), 'w').close()

for seed in gold_seed_idxs:

    print(tune_count+1)
    tune_count +=1

    random.seed(2)

    for gen_count in range(10):
        genTune(seed, gen_count, len(gold_full[j]))

    j+=1
#Generating tunes for multi-blue
'''
