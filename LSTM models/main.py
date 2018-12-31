import argparse
import torch
import pickle
from utils import data_test
from utils import process
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Sentence Completion Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model_bi', type=str, default='./models/model.pt',
                    help='model checkpoint to use')
parser.add_argument('--dict', type=str, default='./dictionary/dict.pt',
                    help='path to pickled dictionary')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--file', type=str, default='./data/file.txt',
                    help='use when giving inputs through file instead of STDIN')
parser.add_argument('--file_fill', type=str, default='./data/file_fill.txt',
                    help='use when giving inputs through file instead of STDIN')
parser.add_argument('--N', type=int, default=3,
                    help='denotes number of words displayed (top N words predicted are displayed)')
parser.add_argument('--sen_length', type=int,
                    default=50,
                    help='Threshold for limiting sentences of the data '
                         '(to restrict unnecessary long sentences)')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

with open(args.model_bi, 'rb') as f:
        model = torch.load(f, map_location = device)
model.eval()


dictionary, threshold = pickle.load(open(args.dict, "rb"))
ntokens = len(dictionary)


def complete_sentence(sentence, index):
    left_ids, right_ids = data_test.tokenize_input(sentence, dictionary)
  
    hidden_left = model.init_hidden(1)
    hidden_right = model.init_hidden(1)
    input_left = torch.LongTensor(left_ids).view(-1, 1).to(device)
    input_right = torch.LongTensor(right_ids).view(-1, 1).to(device)

    outputs = model.text_imputation(input_left, input_right, hidden_left, hidden_right)
    output_flat = outputs.view(-1, ntokens)[-1]  # check this

    missing_word = process.get_missing_word(output_flat, dictionary, args.N)

    print("Candidate words (bidirectional-model): \t\t", end="")
    missing = process.print_predictions(dictionary, missing_word)


    fig, ax = plt.subplots()
    sentence = sentence.replace("___", "")

    ax.set_xticks(np.arange(len(sentence.split()) + 2))
    ax.set_xticklabels([x for x in ["<sos>"] + sentence.split() + ["eos"]])

    plt.xticks(rotation="45")

    if index != 0:
        #plt.savefig('Attention_images/{0}.png'.format(index))
        plt.close()
    else:
        plt.show()

    print()

    return missing


if args.file == '#stdin#':

    sentence = input("Enter sentence (Enter $TOP to stop)\n")
    while sentence != "$TOP":
        try:
            complete_sentence(sentence, 0)
        except Exception as e:
            print(e)

        sentence = input("Enter sentence (Enter $TOP to stop)\n")

else:
    with open(args.file_fill, "w") as out:
        #out.write('id,"sentence"')
        with open(args.file, "r") as f:
            index = 0
            for line in f:
                index += 1
                print(str(index)+". "+line, end="")
                try:
                    missing = complete_sentence(line, index)
                    line = line.replace('___',missing)
                    out.write(line)

                except Exception as e:
                    print(e)
