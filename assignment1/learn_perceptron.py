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

    plot_perceptron.plot_perceptron(neg_example, pos_example, mistakes0, mistakes1,
                    num_err_history, w, w_dist_history)

    # control the procedure
    key = input('<Press enter to continue, q to quit.>>')
    if key == 'q':
        return

    # If a generously feasible weight vectors exists, record the distance
    # to it from the initial weight vector
    if len(w_gen_feas) != 0:
        w_dist_history.append(np.linalg.norm(w-w_gen_feas))

    # iterate untile the perceptron has correctly classified all data points
    while num_errs > 0:
        iteration += 1

        # update the weights of the perceptron
        w = update_weights(neg_example, pos_example, w)
        print(w)


def update_weights(neg_examples, pos_examples, w_current):
    w = w_current
    num_neg_examples = len(neg_examples)
    num_pos_examples = len(pos_examples)

    for element in neg_examples:
        activation = np.dot(element,w)
        key = input()
        if activation >= 0:
            w -= np.transpose(np.array(element))
    for element in pos_examples:
        activation = np.dot(element,w)
        if activation < 0:
            w += np.transpose(element)
    return w


def eval_perceptron(neg_examples, pos_examples, w):
    num_neg_examples = len(neg_examples)
    num_pos_examples = len(pos_examples)

    mistakes0 = []
    mistakes1 = []

    for index, element in enumerate(neg_examples):
        if (np.dot(element,w) > 0):
            mistakes0.append(index)

    for index, element in enumerate(pos_examples):
        if (np.dot(element,w) < 0):
            mistakes1.append(index)

    return(mistakes0, mistakes1)
