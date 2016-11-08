import numpy as np
from load_data import load_data
from fprop import fprop

def train(epochs):

    # set hyperparameters here
    batchsize = 100
    learning_rate = 0.1
    momentum = 0.9
    numhid1 = 50
    numhid2 = 200
    init_wt = 0.01

    # varialbes for tracking training progress
    show_training_CE_after = 100
    show_validation_CE_after = 1000

    # load data
    [train_input, train_target, valid_input,valid_target
     , test_input, test_target, vocab] = load_data(batchsize)
    [numwords, batchsize, numbatches] = train_input.shape
    vocab_size = len(vocab)

    # initialize weighs and bias
    word_embedding_weights = init_wt * np.random.randn(vocab_size,numhid1)
    embed_to_hid_weights = init_wt * np.random.randn(numwords*numhid1,numhid2)
    hid_to_outpout_weights = init_wt * np.random.randn(numhid2,vocab_size)
    hid_bias = np.zeros((numhid2,1))
    output_bias = np.zeros((vocab_size,1))

    word_embedding_weights_delta = np.zeros((vocab_size,numhid1))
    word_embedding_weights_gradient = np.zeros((vocab_size,numhid1))
    embed_to_hid_weights_delta = np.zeros((numwords*numhid1,numhid2))
    hid_to_outpout_weights_delta = np.zeros((numhid2,vocab_size))
    hid_bias_delta = np.zeros((numhid2,1))
    output_bias_delta = np.zeros((vocab_size,1))
    expansion_matrix = np.eye(vocab_size)
    count = 0
    tiny = np.exp(-30)

    for epoch in range(epochs):
        print("1 Epoch {}".format(epoch))
        this_chunk_CE = 0
        train_CE = 0

        # loop over minibatch
        for m in range(numbatches):
            input_batch = train_input[:,:,m]
            target_batch = train_target[:,:,m]

            # forward propagate
            # compute the state of each layer in the network given
            # the input batch and all weights and biases
            [embedding_layer_state, hidden_layer_state,output_layer_state] = fprop(
                input_batch, word_embedding_weights,embed_to_hid_weights,
                hid_to_outpout_weights, hid_bias, output_bias)






if __name__ == "__main__":
    train(10)