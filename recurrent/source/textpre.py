import nltk
import itertools
import numpy as np


def textpre_w2v(path):
    # Initialize variables and parameters
    sentences = []
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"
    unknown_token = "UNKNOWN_TOKEN"
    vocabulary_size = 6000

    file_speech = open(path)

    # tokenize the sentence
    sentences = itertools.chain(*[nltk.sent_tokenize(line.decode('utf-8').lower())
                                                     for line in file_speech])
    sentences = ["%s %s %s" % (sentence_start_token,line,sentence_end_token)
                 for line in sentences]
    print " %d sentences parsed" % len(sentences)

    # word tokenize
    tokenize_sentence = [nltk.word_tokenize(sent) for sent in sentences]
    #print tokenize_sentence
    word_freq = nltk.FreqDist(itertools.chain(*tokenize_sentence))
    print "%d unique words have been found" % len(word_freq.items())

    # get the most common vocabulary dictionary to build word to vector
    vocab = word_freq.most_common(vocabulary_size)
    word_to_index = [w[0] for w in vocab]
    word_to_index.append(unknown_token)
    word_to_vector = dict([(w,i) for i,w in enumerate(word_to_index)])

    ## replace all the word in speech to index
    for i,sent in enumerate(tokenize_sentence):
        tokenize_sentence[i] = [w if w in word_to_index else unknown_token for w in sent]

    # create the train data
    train_x = np.asarray([[word_to_vector[w] for w in sent[:-1]] for sent in tokenize_sentence])
    train_y = np.asarray([[word_to_vector[w] for w in sent[1:]] for sent in tokenize_sentence])
    return train_x, train_y, word_to_vector

if __name__ == "__main__":
    x, y, dict_my = textpre_w2v('../data/speeches.txt')
    print x[0]