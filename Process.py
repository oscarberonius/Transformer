import pandas as pd
import torchtext
from torchtext import data
from Tokenize import tokenize
from Batch import MyIterator, batch_size_fn
import os
import dill as pickle
import random

def read_data(opt):
    
    if opt.src_data is not None:
        try:
            opt.src_data = open(opt.src_data).read().strip().split('\n')
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()
    
    if opt.trg_data is not None:
        try:
            opt.trg_data = open(opt.trg_data).read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()

def create_fields(opt):
    
    spacy_langs = ['en', 'fr', 'de', 'es', 'pt', 'it', 'nl']
    if opt.src_lang not in spacy_langs:
        print('invalid src language: ' + opt.src_lang + 'supported languages : ' + spacy_langs)  
    if opt.trg_lang not in spacy_langs:
        print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + spacy_langs)
    
    print("loading spacy tokenizers...")
    
    t_src = tokenize(opt.src_lang)
    t_trg = tokenize(opt.trg_lang)

    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()
        
    return(SRC, TRG)

def create_dataset(opt, SRC, TRG, val_cutoff):
    print("creating dataset and iterator... ")

    zipped = list(zip([line for line in opt.src_data], [line for line in opt.trg_data]))

    random.shuffle(zipped)

    raw_src, raw_trg = zip(*zipped)
    cutoff = int(len(raw_src)*val_cutoff)

    raw_train = {'src' : raw_src[0:cutoff], 'trg' : raw_trg[0:cutoff]}
    raw_val = {'src' : raw_src[cutoff::], 'trg' : raw_trg[cutoff::]}

    df_train = pd.DataFrame(raw_train, columns=["src", "trg"])
    df_val = pd.DataFrame(raw_val, columns=["src", "trg"])

    mask_train = (df_train['src'].str.count(' ') < opt.max_strlen) & (df_train['trg'].str.count(' ') < opt.max_strlen)
    mask_val = (df_val['src'].str.count(' ') < opt.max_strlen) & (df_val['trg'].str.count(' ') < opt.max_strlen)

    df_train = df_train.loc[mask_train]
    df_val = df_val.loc[mask_val]

    df_train.to_csv("translate_transformer_train_temp.csv", index=False)
    df_val.to_csv("translate_transformer_val_temp.csv", index=False)

    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_train_temp.csv', format='csv', fields=data_fields)
    val = data.TabularDataset('./translate_transformer_val_temp.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True, shuffle=True)
    
    val_iter = MyIterator(val, batch_size=opt.batchsize, device=opt.device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True, shuffle=True)
    
    os.remove('translate_transformer_train_temp.csv')
    os.remove('translate_transformer_val_temp.csv')

    if opt.load_weights is None:
        SRC.build_vocab(train)
        TRG.build_vocab(train)
        if opt.checkpoint > 0:
            try:
                os.mkdir("weights")
            except:
                print("weights folder already exists, run program with -load_weights weights to load them")
                quit()
            pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
            pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    
    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']

    opt.train_len = get_len(train_iter)
    opt.val_len = get_len(val_iter)

    return train_iter, val_iter

def get_len(train):

    for i, b in enumerate(train):
        pass
    
    return i
