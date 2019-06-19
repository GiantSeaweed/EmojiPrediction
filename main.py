#coding:utf-8
import jieba
import collections
import nltk
import re
import json
import sys
import chardet
import time
import argparse
import csv

from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model

import collections
import nltk
import numpy as np


reload(sys)
sys.setdefaultencoding('utf8')

emoji_kind_file = 'emoji.data'
stopwords_file = "stopwords582.data"

# train_file       = 'subset/train.data'
# train_label_file = 'subset/train.solution'
# train_processed_file = "subset/train_preprocessed.data"
# test_file        = 'subset/test.data'
# test_processed_file = "subset/test_preprocessed.data"
# words_freq_file  = "subset/words_freq.data"
train_file = 'train.data'
train_label_file = 'train.solution'
train_processed_file = "train_preprocessed.data"
test_file = "test.data"
test_processed_file = "test_preprocessed.data"
words_freq_file = "words_freq.data"
save_model_file = "model2_161220039.h5"

num_train_data = 0
num_emoji = 0
# @param line: a single string
# @purpose: remove url, numbers, space, puncr(en)
def regex_change(line): 
    # remove url
    url_regex = re.compile(r"""
        (https?://)?
        ([a-zA-Z0-9]+)
        (\.[a-zA-Z0-9]+)
        (\.[a-zA-Z0-9]+)*
        (/[a-zA-Z0-9]+)*
        """, re.VERBOSE|re.IGNORECASE)

    #remove numbers
    decimal_regex = re.compile(r"[0-9]+")
    #remove space
    space_regex = re.compile(r"\s+")
    #remove punct
    punct_regex = re.compile(r"[\s\.\!\_\?\^\*\(\)\"\',~$%@#&<>`]+")

    line = decimal_regex.sub(r"", line) # remove the digitals first!
    line = url_regex.sub(r"", line)
    line = space_regex.sub(r"", line)
    line = punct_regex.sub(r"", line)
    return line

# @param lines:a list of strings
def delete_stopwords(lines, outfile):
    stopwords = []
    all_words = []
    count = 0
    with open(stopwords_file) as f:
        for l in f:
            stopwords.append(l.rstrip())

    with open(outfile, 'w+') as f:
        for line in lines:
            cut_line = jieba.cut(line, cut_all=False)
            cut_str = " ".join(cut_line)
            final_str = ""
            for word in cut_str.split(" "):
                if word not in stopwords:
                    all_words.append(word) # count the freq
                    final_str += word + " "
                    
            f.write(final_str.rstrip()+"\n")

            count += 1
            if count % 500 == 0:
                print("Delete stopwords in %d lines" % (count))
    
    dict_words = dict(collections.Counter(all_words))
    return dict_words

def preprocess_traindata():
    train_txt_list = []
    train_label_list = []
    with open(train_file) as f:
        for line in f:
            train_txt_list.append(line.rstrip())
    with open(train_label_file) as f:
        for line in f:
            train_label_list.append(line.rstrip()[1:-1])

    for i in range(len(train_txt_list)):
        train_txt_list[i] = regex_change(train_txt_list[i])
    print("Have removed by regex...")
    #去除停用词，并返回词袋字典
    bow_words = delete_stopwords(train_txt_list, train_processed_file)
    print("Have removed the stopwords...")
    #对词袋字典进行排序
    sorted_bow = sorted(bow_words.items(), key=lambda d:d[1], reverse=True)
    print("Have sorted by the freq...")
    with open(words_freq_file, "w+") as output_file:
        json.dump(sorted_bow, output_file, ensure_ascii=False)
        print("Loaded the data!")
    print("We have %d training sentences..." % (len(train_txt_list)) )
    print("We have %d training labels......" % (len(train_label_list)))


def get_processed_testdata(opt):
    test_txt_list = []
    with open(test_file) as f:
        for line in f:
            test_txt_list.append(line.strip().split("\t")[1].rstrip())
    for i in range(len(test_txt_list)):
        test_txt_list[i] = regex_change(test_txt_list[i])
    print("[Test Data]: Have removed by regex...")
    #去除停用词，并返回词袋字典
    bow_words = delete_stopwords(test_txt_list, test_processed_file)
    print("[Test Data]: Have removed the stopwords...")


    filter_words = filter_word_freq(opt.freq_threshold)
    word2idx = {x[0].encode('utf8'): i+2 for i, x in enumerate(filter_words)}
    word2idx["PAD"] = 0
    word2idx["UNK"] = 1

    num_test_data = len(open(test_file).readlines())
    X = np.empty(num_test_data, dtype=list)
    with open(test_processed_file) as f:
        i = 0
        for line in f:
            sentence = line.strip().split(" ")
            seqs = []
            for word in sentence:
                if word in word2idx:
                    seqs.append(word2idx[word])
                else:
                    seqs.append(word2idx["UNK"])
            X[i] = seqs
            i = i + 1
    
    X = sequence.pad_sequences(X, maxlen=opt.max_sentence_len)
    print(X)
    return X

def get_words_info():
    maxlen = 0
    words_freq =  collections.Counter()
    len_list = []
    with open(train_processed_file) as f:
        for line in f:
            sentence = line.strip().split(" ")
            len_list.append(len(sentence))
            if len(sentence) > maxlen:
                maxlen = len(sentence)
        print("max_len  : " + str(maxlen)) 

        for i in range(0, 130 ,10):
            len_cut = [item for item in len_list if item > i]
            print("len > %d : %d" % (i, len(len_cut)))
    with open(words_freq_file) as f:
        words_set = json.load(f)
        print("nb_words : " + str(len(words_set)) )
        for i in range(1, 500, 10):
            words_cut_set = [item for item in words_set if item[1] > i]
            print("nb_words > %d : %d" % (i, len(words_cut_set)))

def filter_word_freq(threshold):
    with open(words_freq_file) as f:
        words_set = json.load(f)
        # print("nb_words : " + str(len(words_set)) )
        # for i in range(1, 1000, 50):
        words_cut_set = [item for item in words_set if item[1] > threshold]
        # print("nb_words > %d : %d" % (threshold, len(words_cut_set)))
        return words_cut_set

def main(opt):
    # preprocess_traindata()
    # assert False
    get_words_info()
    num_train_data = len(open(train_file).readlines())
    num_emoji      = len(open(emoji_kind_file).readlines())

    filter_words = filter_word_freq(opt.freq_threshold)
    vocal_size = min(opt.max_vocal_size, len(filter_words)) + 2

    word2idx = {x[0].encode('utf8'):i+2 for i,x in enumerate(filter_words)}
    word2idx["PAD"] = 0
    word2idx["UNK"] = 1
    idx2word = {v:k for k,v in word2idx.items() }

    emoji2idx = {}
    with open("emoji.data") as f:
        i = 0
        for line in f:
            e = line.rstrip().split("\t")[1]
            emoji2idx[e] = i
            i = i+1
    idx2emoji = {v: k for k, v in emoji2idx.items()}

    X = np.empty(num_train_data, dtype = list)
    y = np.zeros(num_train_data, dtype = int)

    with open(train_processed_file) as f:
        i = 0
        for line in f:
            sentence = line.strip().split(" ")
            seqs = []
            for word in sentence:
                if word in word2idx:
                    seqs.append(word2idx[word])
                else:
                    seqs.append(word2idx["UNK"])
            X[i] = seqs
            i = i + 1
    with open(train_label_file) as f:
        i = 0
        for line in f:
            label = line[line.find('{')+1 : line.find('}')]
            #  print(len(emoji2idx[label]))
            label.decode('utf-8')
            y[i] = emoji2idx[label]
            i = i + 1

    X = sequence.pad_sequences(X, maxlen = opt.max_sentence_len )
    y = np_utils.to_categorical(y, num_emoji)
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, 
                                    test_size=0.001, random_state=42)

    #########################################
    # construct the model
    #########################################
    model = Sequential()
    # the model will take as input an int matrix of size(batch, input_length).
    # the largest int(word index) in the input should be no larger than vocal_size.
    # now model.output_shape == (None, input_length, embedding_size), where None is the batch dimension.
    model.add(Embedding(vocal_size, 
                        opt.embedding_size, 
                        input_length = opt.max_sentence_len))
    # return_sequence = False(default)
    model.add(LSTM(opt.hidden_layer_size, dropout=opt.dropout, recurrent_dropout=opt.rec_dropout))
    model.add(Dense(num_emoji))
    model.add(Activation('softmax'))


    plot_model(model, to_file='model.png',
              show_shapes=True, show_layer_names=False)



    print("Compiling the model....")
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam', metrics = ["accuracy"])
    #########################################
    # train the model
    #########################################
    model.fit(Xtrain, ytrain, 
              batch_size = opt.batch_size,
              epochs = opt.epoch, validation_data=(Xvalid, yvalid))
    
    score, acc = model.evaluate(Xvalid, yvalid, batch_size=opt.batch_size)
    print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))

    #########################################
    # predict on the real test data
    #########################################
    print("\nPredicting on the test data....")
    Xtest = get_processed_testdata(opt)
    y_predict = model.predict(Xtest)
    
    submission_file = "embed" + str(opt.embedding_size) +"_hid"+ str(opt.hidden_layer_size) \
                        + "_batch" + str(opt.batch_size) \
                        + "_epoch" + str(opt.epoch) \
                        + "_thrsh" + str(opt.freq_threshold) \
                        + "_maxlen" + str(opt.max_sentence_len) \
                        + "_dp" + str(opt.dropout) \
                        + "_rdp" + str(opt.rec_dropout)

    with open(submission_file, "w+") as f:
        w = csv.writer(f)
        w.writerow(["ID","Expected"])
        for i in range(len(y_predict)):
            index = y_predict[i].tolist().index(max(y_predict[i]))
            w.writerow([str(i), str(index)])
    print("\nWrite the predicting results into the csv! ")

    model.save(save_model_file)
    print("Save the model!")

if __name__ == "__main__":
    start = time.time()
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument('-gpuid', type=int, default=1, help='which gpu to use. -1 = use CPU')
    main_arg_parser.add_argument('-embedding_size', type=int, default=300, help='word embedding size')
    main_arg_parser.add_argument('-hidden_layer_size', type=int, default=32, help='hidden layer size')
    main_arg_parser.add_argument(
        '-max_vocal_size', type=int, default=150000, help='max vocal size')
    main_arg_parser.add_argument('-max_sentence_len', type=int, default=110, help='max sentence len')
    main_arg_parser.add_argument(
        '-batch_size', type=int, default=32, help='batch size')
    main_arg_parser.add_argument(
        '-epoch', type=int, default=1, help='epoch num')
    main_arg_parser.add_argument(
        '-freq_threshold', type=int, default=5, help='freq threshold')
    main_arg_parser.add_argument(
        '-dropout', type=float, default=0.2, help='dropout')
    main_arg_parser.add_argument(
        '-rec_dropout', type=float, default=0.2, help='recurrent dropout')

    
    args = main_arg_parser.parse_args()
    main(args)
