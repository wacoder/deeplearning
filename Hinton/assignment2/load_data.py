import scipy.io as sio
import numpy as np
import math

def load_data(N):
    data = sio.loadmat('./data/data.mat')['data']

    # convert the ndarray to array (here read from
    # .mat has some bugs for shape of ndarray)
    vocab = data['vocab'][0][0][0]
    trainData = data['trainData'][0][0]
    validData = data['validData'][0][0]
    testData = data['testData'][0][0]

    # reshape the traindata
    numdims = trainData.shape[0]
    M = math.floor(trainData.shape[1]/N)
    D = numdims - 1

    train_input =   np.array(trainData[0:D,0:N*M]).reshape((D,N,M))
    train_target = np.array(trainData[D,0:N*M]).reshape((1,N,M))

    valid_input = np.array(validData[0:D,:])
    valid_target = np.array(validData[D,:])

    test_input = np.array(testData[0:D,:])
    test_target = np.array(testData[D,:])

    return(train_input, train_target, valid_input, valid_target,
           test_input, test_target, vocab)



if __name__ == "__main__":
    [train_input, train_target, valid_input, valid_target,test_input, test_target, vocab] = load_data(100)