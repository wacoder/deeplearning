import numpy as np

def softmax(x):

    # exponential of the input
    exp_x = np.exp(x)
    return exp_x/sum(exp_x)



if __name__ == "__main__":
    x = [1,2,3]
    print softmax(x)