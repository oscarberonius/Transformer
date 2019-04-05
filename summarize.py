import argparse
import time
import torch
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import pdb
import dill as pickle
import argparse
from Models import get_model
from Beam import beam_search
from nltk.corpus import wordnet
from torch.autograd import Variable
import re
import pandas as pd

def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]
            
    return 0

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

def summarize_sentence(sentence, model, opt, SRC, TRG):
    
    model.eval()
    indexed = []
    sentence = SRC.preprocess(sentence)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 or opt.floyd == True:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = Variable(torch.LongTensor([indexed]))
    if opt.device == 0:
        sentence = sentence.cuda()

    sentence, score = beam_search(sentence, model, SRC, TRG, opt)

    return  multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, sentence), score

def summarize(opt, model, SRC, TRG):
    paragraph = opt.text.lower()
    return summarize_sentence(paragraph, model, opt, SRC, TRG)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=150)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-src_lang', default='en')
    parser.add_argument('-trg_lang', default='en')
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    
    opt = parser.parse_args()

    opt.device = 0 if opt.no_cuda is False else -1
 
    assert opt.k > 0
    assert opt.max_len > 10

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    
    while True:
        opt.text =input("Enter a paragraph to summarize (type 'f' to load from file, or 'q' to quit):\n")
        if opt.text=="q":
            break
        if opt.text=='f':
            fpath =input("Enter a paragraph to summarize (type 'f' to load from file, or 'q' to quit):\n")
            try:
                opt.text = ' '.join(open(opt.text, encoding='utf-8').read().split('\n'))
            except:
                print("error opening or reading text file")
                continue
        phrase, score = summarize(opt, model, SRC, TRG)
        score = score.data.cpu().numpy()
        print('> '+ phrase + " | Score: "+score+'\n')
def generate_result_data():

    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=120)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-src_lang', default='en')
    parser.add_argument('-trg_lang', default='en')
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')

    opt = parser.parse_args()

    opt.device = 0 if opt.no_cuda is False else -1

    assert opt.k > 0
    assert opt.max_len > 10

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    #with open('raw_train_420kdps_gd.json', 'rb') as f:
    #    raw_train = json.load(f)
    
    with open('raw/raw_val.json', 'rb') as f:
        raw_val = json.load(f)

    #df_train = pd.DataFrame(raw_train, columns=["src", "trg"])
    df_val = pd.DataFrame(raw_val, columns=["src", "trg"])
    
    #mask_train = (df_train['src'].str.count(' ') < opt.max_len) & (df_train['trg'].str.count(' ') < opt.max_len)
    mask_val = (df_val['src'].str.count(' ') < opt.max_len) & (df_val['trg'].str.count(' ') < opt.max_len)

    #df_train = df_train.loc[mask_train]
    df_val = df_val.loc[mask_val]
    # Calculate score from 10000 dps of both training and validation sets. Save to files for later plotting.
    # Save file should be a dict from which it is possible to extract some n top and bottom scores and their corresponding outputs.
    #count = 0
    #train_results = []
    #for dp in df_train.itertuples():
    #    src_dp = dp.src

    #    phrase, score = summarize(opt, model, SRC, TRG)        
    #    result = {'src':src_dp,'trg':trg_dp,'output':phrase,'score':score.item() }
    #    train_results.append(result)

    #    count +=1
    #    if count%100==0:
    #        print(f'Processed datapoints: {count}')
    #    if count >= 2000:
    #        print('Training set tested.')
    #        break
    
    count = 0
    val_results = []
    for dp in df_val.itertuples():
        src_dp = dp.src
        trg_dp = dp.trg
        opt.text = src_dp

        phrase, score = summarize(opt, model, SRC, TRG)
        result = {'src':src_dp,'trg':trg_dp,'output':phrase,'score':score.item()}
        val_results.append(result)

        count +=1
        if count%100==0:
            print(f'Processed datapoints: {count}')
        if count >= 2000:
            print('Validation set tested.')
            break
    
    #with open('train_results_420kdps_gd.json', 'w') as f:
    #    json.dump(train_results, f)
    
    with open('val_results.json', 'w') as f:
        json.dump(val_results, f)
    print('Result data generated')

if __name__ == '__main__':
    #main()
    generate_result_data()
