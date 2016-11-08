import numpy as np

def fprop(input_batch, word_embedding_weights, embed_to_hid_weights,
          hid_to_output_weights, hid_bias, output_bias):
    # This method forward propagates through neural network

    [numwords, batchsize] = input_batch.shape
    [vocab_size, numhid1] = word_embedding_weights.shape
    numhid2 = embed_to_hid_weights.shape[1]


    # compute the state of word embedding layer
    embedding_layer_state = word_embedding_weights[np.array(input_batch).flatten('A')-1,:].flatten('A')
    print(word_embedding_weights[np.array(input_batch).flatten('A')-1,:][0,:])
    print(input_batch)
    print(np.array(input_batch).flatten('A')-1)
