import numpy as np
from utils import softmax

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


if __name__ == "__main__":
    x = np.asarray([[1,0,0],[0,1,0]])
    model = RNNnp(3)
    output = model.predict(x)
    print output



