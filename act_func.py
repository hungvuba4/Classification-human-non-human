import numpy as np
# from scipy.special import expit, softmax


def sig():
    '''sigmoid function'''
    return lambda x: 1 / (1 + np.exp(-x))
# def sig(x):
#    return 1 / (1 + np.exp(-x))


def d_sig():
    '''derivative of sigmoid function'''
    return lambda x: sig()(x) * (1 - sig()(x))
# def d_sig(x):
#     return sig(x) * (1 - sig(x))


def tanh():
    '''tanh function'''
    return lambda x: (np.exp(-x) - np.exp(x)) / (np.exp(-x) + np.exp(x))
# def tanh(x):
#     return (np.exp(-x) - np.exp(x)) / (np.exp(-x) + np.exp(x))


def d_tanh():
    '''derivate of tanh function'''
    return lambda x: 2 * sig()(2 * x) - 1
# def d_tanh(x):
#     return 2 * sig(2 * x) - 1


def relu():
    '''ReLU function'''
    return lambda x: np.where(x > 0, x, 0)
# def relu(x):
#     return np.max(0, x)


def d_relu():
    '''derivate of relu function'''
    return lambda x: np.where(x > 0, 1, 0)
# def d_relu(x):        
#     x[x <= 0] = 0
#     x[x > 0] = 1
#     return x


def softmax():
    '''softmax function'''
    return lambda x: np.devide(np.exp(x), np.exp(x).np.sum(x))

# derivate of softmax function
# pass


def ide():
    '''identity function'''
    return lambda x: x
# def ide(x):
#    return x


def d_ide():
    '''derivate of identity function'''
    return 1


def get_act(act):
    act = act.lower()
    if act == 'sigmoid':
        return sig(), d_sig()
    elif act == 'tanh':
        return tanh(), d_tanh()
    elif act == 'relu':
        return relu(), d_relu()
    elif act == 'identity':
        return ide(), d_ide()
    else:
        return sig(), d_sig()

