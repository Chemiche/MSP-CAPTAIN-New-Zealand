import sys
import numpy as np


def get_rnd_gen(seed=None):
    return np.random.default_rng(seed)


def print_update(s):
    sys.stdout.write('\r')
    sys.stdout.write(s)
    sys.stdout.flush()

def get_nn_params_from_file(logfile, load_best_epoch=False):
    head = next(open(logfile)).split()
    loaded_ws = np.loadtxt(logfile, skiprows=1)
    if load_best_epoch:
        selected_epoch = np.argmax(loaded_ws[:, head.index("running_reward")])
    else:
        selected_epoch = -1
    print(
        "Selected epoch",
        selected_epoch,
        loaded_ws[:, head.index("reward")][selected_epoch],
        loaded_ws[:, head.index("running_reward")][selected_epoch],
    )
    loadedW = loaded_ws[selected_epoch]
    ind = [head.index(s) for s in head if "coeff_" in s]
    wNN = loadedW[np.min(ind):]
    return wNN
