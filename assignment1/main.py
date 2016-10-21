import learn_perceptron
import scipy.io as sio

if __name__ == '__main__':
    data = sio.loadmat('./data/dataset1.mat')
    learn_perceptron.learn_perceptron(data['neg_examples_nobias'], data['pos_examples_nobias'],
                                      data['w_init'], data['w_gen_feas'])