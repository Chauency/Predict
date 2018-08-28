import torch
import numpy as np
import pandas as pd


def reg_l2_fc(m):
    """
    Compute recursively and return the l2 regularization term of given module,
    with consideration of only fully-connected layers.
    """
    sos = 0  # sum of squares

    if m.__class__.__name__ == 'Linear':
        sos += torch.sum(m.weight.pow(2))
    else:
        for child in m.children():
            sos += reg_l2_fc(child)

    return sos


def weights_init(m):
    name = m.__class__.__name__

    if name == 'Linear':
        weight_shape = list(m.weight.shape)
        fan_in = weight_shape[1] * 1.
        fan_out = weight_shape[0] * 1.
        w_bound = 6 / np.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-w_bound, w_bound)

        m.bias.data.fill_(0)

    elif name == 'LSTMCell':
        m.bias_ih.data.fill_(0)
        m.bias_hh.data.fill_(0)


def test_error_writer(name, num_epochs, num_batches, q):
    """
    helper function for subprocess that write the test errors
    """
    for epoch in range(num_epochs):
        with open(f'{name}_e{epoch+1}.csv', mode='w') as f:
            f.write('Test_Err\n')

        for i in range(num_batches):
            loss = q.get()
            df = pd.DataFrame({0: loss})
            df.to_csv(f'{name}_e{epoch+1}.csv', mode='a',
                      index=False, header=False)
