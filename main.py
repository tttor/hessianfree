#!/usr/bin/env python3

import numpy as np
import hessianfree as hf

def main():
    xor(use_hf=True)

def xor(use_hf=True):
    """Run a basic xor training test.
    :param bool use_hf: if True run example using Hessian-free optimization,
        otherwise use stochastic gradient descent
    """
    # data
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    # net
    shape = [2, 5, 1]
    ff = hf.FFNet(shape,
                  layers=hf.nl.Logistic(),
                  conns = None, # default is to connect in series
                  loss_type=hf.loss_funcs.SquaredError(),
                  W_init_params=None,
                  use_GPU=False,
                  load_weights=None,
                  debug=False,
                  rng=np.random.RandomState(12345),
                  dtype=np.float32)

    # train
    if use_hf:
        ff.run_epochs(inputs, targets,
                      optimizer=hf.opt.HessianFree(CG_iter=2),
                      max_epochs=40, plotting=True)
    else:
        # using gradient descent (for comparison)
        ff.run_epochs(inputs, targets, optimizer=hf.opt.SGD(l_rate=1),
                      max_epochs=100, plotting=True)

    # test
    print('### testing...')
    outputs = ff.forward(inputs)[-1]
    for i in range(4):
        print("-" * 2)
        print("input", inputs[i])
        print("target", targets[i])
        print("output", outputs[i])

if __name__ == '__main__':
    main()
