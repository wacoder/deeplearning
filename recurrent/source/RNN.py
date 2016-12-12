import numpy as np
from utils import softmax
from textpre import textpre_w2v
class RNNnp:

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):

        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # Randomly initialize the network parameters
        self.U = np.random.uniform(-1/np.sqrt(word_dim),1/np.sqrt(word_dim),
                              (hidden_dim,word_dim))
        self.W = np.random.uniform(-1/np.sqrt(hidden_dim),1/np.sqrt(hidden_dim),
                              (hidden_dim,hidden_dim))
        self.V = np.random.uniform(-1/np.sqrt(hidden_dim),1/np.sqrt(hidden_dim),
                              (word_dim,hidden_dim))

    def forward_prop(self,x):
        # The total number of word number
        text_len = len(x)

        # initialize the state and output value
        s = np.zeros((text_len+1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        o = np.zeros((text_len, self.word_dim))

        # for each the epoch
        for t in range(text_len):
            # print np.transpose(self.U[:,x[t]!=0])[0]
            # print self.W.dot(s[t-1])
            s[t] = np.tanh(np.transpose(self.U[:,x[t]!=0])[0] + np.dot(self.W,s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))

        return o,s

    def predict(self,x):
        o,s = self.forward_prop(x)
        return np.argmax(o,axis=1)

    def calculate_loss(self, x, y):
        L = 0

        for i in range(len(y)):
            o, s = self.forward_prop(x[i])
            correct_prediction = o[range(len(y[i])), y[i]]

            L -= np.sum(np.log(correct_prediction))
        return L

    def calculate_total_loss(self,x,y):
        N = np.sum(len(y_i) for y_i in y)
        return self.calculate_loss(x,y)/N



if __name__ == "__main__":
    np.random.seed(10)
    x, y, dict_my = textpre_w2v('../data/speeches.txt')
    model = RNNnp(len(dict_my))
    o,s = model.forward_prop(x[10])


    print "Expected loss is %f" %(np.log(len(dict_my)))
    print "Actual loss is {:f}".format(model.calculate_total_loss(x[:1000],y[:1000]))



