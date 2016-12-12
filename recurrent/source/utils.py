import numpy as np

def softmax(x):

    # exponential of the input
    exp_x = np.exp(x)
    return exp_x/sum(exp_x)



if __name__ == "__main__":
    x = [1,2,3]
    d = [[1,2,3,4,5,6],
         [1, 3, 3, 4, 5, 8],
         [1, 2, 3, 4, 5, 6],
         [2, 3, 4, 5, 6, 1]]
    d = np.array(d)
    col = [1,3,4,2]
    print range(len(d))
    print d[(0,1,2,3), (2,1,1,1)]
    print softmax(x)