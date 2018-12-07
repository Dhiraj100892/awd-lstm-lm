import argparse
import time
import math
import numpy as np
import torch
import model
import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling
from IPython import embed
import os
import random
from utils import write_mid_mp3_wav, repackage_hidden, create_paths, load_params


parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=400,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=4,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=200,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='./models/model.pt',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
args = parser.parse_args()
args.tied = True

###############################################################################
# Build the model and load the model
###############################################################################
def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

# first create PATH instance
PATHS = create_paths()
params, TEXT = load_params(PATHS)
ntokens =  params['n_tok']                      # need to load it from param
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
model_load(args.resume)

if args.cuda:
    model = model.cuda()

###############################################################################
# Generation code
###############################################################################
def load_long_prompts(folder):
    """ folder is either the path to train or to test
        load_long_prompts loads all the files in that folder and returns
        a list holding the text inside each file
    """
    prompts=[]
    all_files=os.listdir(folder)
    for i in range(len(all_files)):
        f=open(folder/all_files[i])
        prompt=f.read()
        prompts.append(prompt)
        f.close()
    return prompts

def music_tokenizer(x): return x.split(" ")


def generate_musical_prompts(prompts, bptt, bs):
    prompt_size=bptt
    musical_prompts=[]

    # Randomly select bs different prompts and hold them in musical_prompts
    for i in range(bs):
        this_prompt=[]
        timeout=0
        while timeout<100 and len(this_prompt)-prompt_size<=1:
            sample=random.randint(0,len(prompts)-1)
            this_prompt=prompts[sample].split(" ")
            timeout+=1
        assert len(this_prompt)-prompt_size>1, f'After 100 tries, unable to find prompt file longer than {bptt}. Run with smaller --bptt'

        offset=random.randint(0, len(this_prompt)-prompt_size-1)
        musical_prompts.append(" ".join(this_prompt[offset:prompt_size+offset]))

    return musical_prompts


def create_generation_batch(model, num_words, random_choice_frequency,
                            trunc_size, bs, bptt, prompts, params, TEXT):
    """ Generate a batch of musical samples
    Input:
      model - pretrained generator model
      num_words - number of steps to generate
      random_choice_frequency - how often to pick a random choice rather than the top choice (range 0 to 1)
      trunc_size - for the random choice, cut off the options to include only the best trunc_size guesses (range 1 to vocab_size)
      bs - batch size - number of samples to generate
      bptt - back prop through time - size of prompt
      prompts - a list of training or test folder texts
      params - parameters of the generator model
      TEXT - holds vocab word to index dictionary

    Output:
      musical_prompts - the randomly selected prompts that were used to prime the model (these are human-composed samples)
      results - the generated samples

    This is very loosely based on an example in the FastAI notebooks, but is modified to include randomized prompts,
    to generate a batch at a time rather than a single example, and to include truncated random sampling.
    """
    hidden = model.init_hidden(bs)

    musical_prompts=generate_musical_prompts(prompts, bptt, bs)

    results=['']*bs
    model.eval()

    # Tokenize prompts and translate them to indices for input into model
    s = [music_tokenizer(prompt)[:bptt] for prompt in musical_prompts]
    t=TEXT.numericalize(s)

    print("Prompting network")
    # Feed the prompt one by one into the model (b is a vector of all the indices for each prompt at a given timestep)
    for b in t:
        res, hidden = model(b.unsqueeze(0).cuda(), hidden)

    print("Generating new sample")
    for i in range(num_words):
        res = model.decoder(res)
        # res holds the probabilities the model predicted given the input sequence
        # n_tok is the number of tokens (ie the vocab size)
        [ps, n] =res.topk(params["n_tok"])

        # By default, choose the most likely word (choice 0) for the next timestep (for all the samples in the batch)
        w=n[:,0]

        # Cycle through the batch, randomly assign some of them to choose from the top trunc guesses, rather than to
        # automatically take the top choice
        for j in range(bs):
            """
            if random.random()<random_choice_frequency:
                # Truncate to top trunc_size guesses only
                ps=ps[:,:trunc_size]
                # Sample based on the probability the model predicted for those top choices
                r=torch.multinomial(ps[j].exp(), 1)
                # Translate this to an index
                #TODO: need to figure it out
                ind=to_np(r[0])[0]
                if ind!=0:
                    w[j].data[0]=n[j,ind].data[0]
            """
            # Translate the index back to a word (itos is index to string)
            # Append to the ongoing sample
            results[j]+=TEXT.vocab.itos[w[j].item()]+" "

        # Feed all the predicted words from this timestep into the model, in order to get predictions for the next step
        res, hidden = model(w.unsqueeze(0).cuda(), hidden)
    return musical_prompts,results


## main code
# we are doing notewise prediction
sample_freq = 12
note_offset=33
chordwise = False
generator_bs = 8
gen_size = 2000
bptt = 200
random_freq = 0.5
trunc = 5

# load the model
model_load(args.resume)
model.eval()
hidden = model.init_hidden(generator_bs)

# load the text
prompts=load_long_prompts(PATHS["data"]/"test")
print("Preparing to generate a batch of "+str(generator_bs)+" samples.")
musical_prompts,results=create_generation_batch(model=model, num_words=gen_size,
                                                    bs=generator_bs, bptt=bptt,
                                                    random_choice_frequency=random_freq,
                                                    trunc_size=trunc, prompts=prompts,
                                                    params=params, TEXT=TEXT)

# Create the output folder if it doesn't already exist
out=PATHS["output"]/output_folder
out.mkdir(parents=True, exist_ok=True)

# For each generated sample, write mid, mp3, wav, and txt files to the output folder (as 1.mid, etc)
for i in range(len(results)):
    write_mid_mp3_wav(results[i], str(i).zfill(2)+".mid", sample_freq, note_offset, out, chordwise)
    fname=str(i)+".txt"
    f=open(out/fname,"w")
    f.write(results[i])
    f.close()

# For each human-composed sample, write mid, mp3, and wav files to the output folder (as prompt1.mid, etc)
for i in range(len(musical_prompts)):
    write_mid_mp3_wav(musical_prompts[i], "prompt"+str(i).zfill(2)+".mid", sample_freq, note_offset, out, chordwise)

