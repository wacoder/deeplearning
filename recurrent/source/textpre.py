import nltk
import itertools


def textpre_w2v(path):
    # Initialize variables and parameters
    sentences = []
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"

    file_speech = open(path)


    sentences = itertools.chain(*[nltk.sent_tokenize(line.decode('utf-8').lower())
                                                     for line in file_speech])
    sentences = ["%s %s %s" % (sentence_start_token,line,sentence_end_token)
                 for line in sentences]
    print " %d sentences parsed" % len(sentences)

if __name__ == "__main__":
    textpre_w2v('../data/speeches.txt')