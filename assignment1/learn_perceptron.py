import numpy as np
import plot_perceptron

def learn_perceptron(neg_example_nobias, pos_examples_nobias, w_init, w_gen_feas):
    # the size of input
    num_neg_examples = len(neg_example_nobias)
    num_pos_examples = len(pos_examples_nobias)
    num_err_history = []
    w_dist_history  = []

    neg_example = np.c_[neg_example_nobias, np.ones(num_neg_examples)]
    pos_example = np.c_[pos_examples_nobias, np.ones(num_pos_examples)]

    if 'w_init' not in locals() or w_init.size == 0:
        w = np.random.rand(3,1)
    else:
        w = w_init

    if 'w_gen_feas' in locals():
        w_gen_feas = []

    # find the data points that the perceptron has incorrectly classified
    iteration = 0
    [mistakes0,mistakes1] = eval_perceptron(neg_example,pos_example,w)

    num_errs = len(mistakes0) + len(mistakes1)
    num_err_history.append(num_errs)

    print("Number of errors in iteration %d: \t %d\n"%(iteration, num_errs))
    print("Weights:\t",np.transpose(w),'\n')

    plot_perceptron(neg_example, pos_example, mistakes0, mistakes1,
                    )


def eval_perceptron(neg_examples, pos_examples, w):
    num_neg_examples = len(neg_examples)
    num_pos_examples = len(pos_examples)

    mistakes0 = []
    mistakes1 = []

    for index, element in enumerate(neg_examples):
        if (np.dot(element,w) > 0):
            mistakes0.append(index+1)

    for index, element in enumerate(pos_examples):
        if (np.dot(element,w) < 0):
            mistakes1.append(index+1)

    return(mistakes0, mistakes1)
