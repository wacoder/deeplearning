import matplotlib.pyplot as plt

def plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history):
    f = plt.figure(1)
    plt.clf()

    # get the correct assigned data points
    neg_correct_ind = list(set(range(0,len(neg_examples)))-set(mistakes0))
    pos_correct_ind = list(set(range(1,len(pos_examples)))-set(mistakes1))

    # plot the data points and decision boundary
    plt.subplot(221)

    if neg_examples.size != 0:
        plt.plot(neg_examples[neg_correct_ind,0],neg_examples[neg_correct_ind,1],'og', markersize=10)
    if pos_examples.size != 0:
        plt.plot(pos_examples[pos_correct_ind,0],pos_examples[pos_correct_ind,1],'sg',markersize=10)
    if len(mistakes0) > 0:
        plt.plot(neg_examples[mistakes0,0],neg_examples[mistakes0,1],'or', markersize=10)
    if len(mistakes1) > 0:
        plt.plot(pos_examples[mistakes1,0],pos_examples[mistakes1,1],'sr', markersize=10)

    plt.title("Classifier")
    plt.plot([-5,5],[(-w[-1]+5*w[0])/w[1],(-w[-1]-5*w[0])/w[1]], 'k')
    plt.xlim([-1,1])
    plt.ylim([-1,1])

    # plot the error history
    plt.subplot(222)

    plt.plot(range(len(num_err_history)),num_err_history)
    plt.xlim([-1, max(15, len(num_err_history))])
    plt.ylim([0,len(neg_examples)+len(pos_examples)+1])
    plt.title("Number of errors")
    plt.xlabel('Iteration')
    plt.ylabel('Number of errors')

    # plot the ?
    plt.subplot(223)
    plt.plot(range(len(w_dist_history)),w_dist_history)
    plt.xlim([-1,max(15,len(num_err_history))])
    plt.ylim([0,15])
    plt.title('Distance')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.show()
