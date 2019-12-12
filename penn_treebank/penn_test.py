"""
based on https://github.com/locuslab/TCN/tree/master/TCN/char_cnn
"""

import argparse
import torch.nn as nn
import torch.optim as optim
import time
import math
from RNN_compression.cells import WaveletGRU, FastFoodGRU, GRUCell
from util import compute_parameter_total
from penn_treebank.char_utils import *
import collections
import sys
sys.path.append("../")

CustomWavelet = collections.namedtuple('Wavelet', ['dec_lo', 'dec_hi',
                                                   'rec_lo', 'rec_hi', 'name'])

import warnings
warnings.filterwarnings("ignore")   # Suppress the RunTimeWarning on unicode

parser = argparse.ArgumentParser(description='Sequence Modeling - Character Level Language Model')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--clip', type=float, default=0.15,
                    help='gradient clip, -1 means no clip (default: 0.15)')
parser.add_argument('--epochs', type=int, default=60,
                    help='upper epoch limit (default: 60)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=0.005,
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--emsize', type=int, default=100,
                    help='dimension of character embeddings (default: 100)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: SGD)')
parser.add_argument('--validseqlen', type=int, default=320,
                    help='valid sequence length (default: 320)')
parser.add_argument('--seq_len', type=int, default=400,
                    help='total sequence length, including effective history (default: 400)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--dataset', type=str, default='ptb',
                    help='dataset to use (default: ptb)')
parser.add_argument('--cell', type=str, default='GRU', help='The cell type used')
parser.add_argument('--cell_size', type=int, default=512, help='Cell state size.')
parser.add_argument('--compression_mode', type=str, default='state',
                    help='Where to apply the compression layers.')
parser.add_argument('--wavelet_weight', type=float, default=1.,
                    help='Weight factor for the wavelet loss.')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


print(args)
file, file_len, valfile, valfile_len, testfile, testfile_len, corpus = data_generator(args)

n_characters = len(corpus.dict)
train_data = batchify(char_tensor(corpus, file), args.batch_size, args)
val_data = batchify(char_tensor(corpus, valfile), 1, args)
test_data = batchify(char_tensor(corpus, testfile), 1, args)
print("Corpus size: ", n_characters)


# num_chans = [args.nhid] * (args.levels - 1) + [args.emsize]

print(args.cell)
if args.cell == 'WaveGRU':
    init_wavelet = CustomWavelet(dec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                 dec_hi=[0, 0, -0.7071067811865476, 0.7071067811865476, 0, 0],
                                 rec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                 rec_hi=[0, 0, 0.7071067811865476, -0.7071067811865476, 0, 0],
                                 name='custom')
    cell = WaveletGRU(input_size=args.emsize, out_size=n_characters, hidden_size=args.cell_size,
                      mode=args.compression_mode)
elif args.cell == 'GRU':
    cell = GRUCell(input_size=args.emsize, out_size=n_characters, hidden_size=args.cell_size)
else:
    raise NotImplementedError()

# model = TCN(args.emsize, n_characters, num_chans, kernel_size=k_size, dropout=dropout, emb_dropout=emb_dropout)


class EmbeddingRnnWrapper(torch.nn.Module):
    def __init__(self, cell, input_size, out_size):
        super(EmbeddingRnnWrapper, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.cell = cell
        self.encoder = nn.Embedding(out_size, input_size)

    def forward(self, x, h):
        emb = self.encoder(x)
        return self.cell(emb, h)

    def get_wavelet_loss(self):
        return self.cell.get_wavelet_loss()


model = EmbeddingRnnWrapper(cell, input_size=args.emsize, out_size=n_characters)
parameter_total = compute_parameter_total(model)
print('parameter total', parameter_total)

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def run_rnn(cell, input):
    y_cell_lst = []
    h = None
    for t in range(input.shape[-1]):
        # batch_major format [b,t,d]
        y, h = cell(input[:, t], h)
        y_cell_lst.append(y)

    y = torch.stack(y_cell_lst, 1)
    return y


def evaluate(source):
    model.eval()
    total_loss = 0
    count = 0
    acc_sum = 0
    source_len = source.size(1)
    with torch.no_grad():
        for batch, i in enumerate(range(0, source_len - 1, args.validseqlen)):
            if i + args.seq_len - args.validseqlen >= source_len:
                continue
            inp, target = get_batch(source, i, args)

            # output = model(inp)
            output = run_rnn(model, inp)

            eff_history = args.seq_len - args.validseqlen
            final_output = output[:, eff_history:].contiguous().view(-1, n_characters)
            final_target = target[:, eff_history:].contiguous().view(-1)
            loss = criterion(final_output, final_target)

            total_loss += loss.data * final_output.size(0)
            count += final_output.size(0)

            # compute accuracy.
            acc_sum += torch.sum((torch.max(final_output, -1)[1] == final_target).type(torch.float32))

        val_loss = total_loss.item() / count * 1.0
        val_acc = acc_sum.item() / count * 1.0
        return val_loss, val_acc


def train(epoch):
    model.train()
    total_loss = 0
    total_wvl_loss = 0
    start_time = time.time()
    losses = []
    wvl_losses = []
    source = train_data
    source_len = source.size(1)
    for batch_idx, i in enumerate(range(0, source_len - 1, args.validseqlen)):
        if i + args.seq_len - args.validseqlen >= source_len:
            continue
        inp, target = get_batch(source, i, args)
        optimizer.zero_grad()

        # output = model(inp)
        output = run_rnn(model, inp)

        eff_history = args.seq_len - args.validseqlen
        final_output = output[:, eff_history:].contiguous().view(-1, n_characters)
        final_target = target[:, eff_history:].contiguous().view(-1)
        criterion_loss = criterion(final_output, final_target)

        if args.cell == 'WaveGRU':
            loss_wave = model.get_wavelet_loss()
            loss = criterion_loss + loss_wave * args.wavelet_weight
            # print(loss_wave.item())
        else:
            loss_wave = torch.tensor(0.)
            loss = criterion_loss

        loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += criterion_loss.item()
        total_wvl_loss += loss_wave.item()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / args.log_interval
            cur_wvl_loss = total_wvl_loss / args.log_interval
            losses.append(cur_loss)
            wvl_losses.append(cur_wvl_loss)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                  'loss {:5.3f} | bpc {:5.3f}'.format(
                epoch, batch_idx, int((source_len-0.5) / args.validseqlen), lr,
                              elapsed * 1000 / args.log_interval, cur_loss, cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()

    return sum(losses) * 1.0 / len(losses), sum(wvl_losses)/len(wvl_losses)


def main():
    global lr
    print("Training for %d epochs..." % args.epochs)
    all_losses = []
    best_vloss = 1e7
    for epoch in range(1, args.epochs + 1):
        loss, wvl_loss = train(epoch)
        print('| epoch {:3d} | loss {:5.3f} | bpc {:8.3f} | wvl-loss {:8.3f}'.format(
              epoch, loss, loss / math.log(2), wvl_loss))
        vloss, vacc = evaluate(val_data)
        print('-' * 89)
        print('| End of epoch {:3d} | valid loss {:5.3f} | valid bpc {:8.3f}| valid acc {:8.3f}'.format(
            epoch, vloss, vloss / math.log(2), vacc))

        test_loss, test_acc = evaluate(test_data)
        print('=' * 89)
        print('| End of epoch {:3d} | test loss {:5.3f} | test bpc {:8.3f}  | test acc {:8.3f}'.format(
            epoch, test_loss, test_loss / math.log(2), test_acc))
        print('=' * 89)

        if epoch > 5 and vloss > max(all_losses[-3:]):
            lr = lr / 2.
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        all_losses.append(vloss)

        # if vloss < best_vloss:
        #     print("Saving...")
        #     save(model)
        #     best_vloss = vloss

    # Run on test data.
    test_loss, test_acc = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.3f} | test bpc {:8.3f}'.format(
        test_loss, test_loss / math.log(2)))
    print('=' * 89)
    print('parameter total', parameter_total)


# train_by_random_chunk()
if __name__ == "__main__":
    main()
