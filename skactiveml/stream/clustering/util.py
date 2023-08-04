from multiprocessing import Pool
from time import sleep
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def save_image(filename):
    # PdfPages is a wrapper around pdf
    # file so there is no clash and
    # create files with no error.
    p = PdfPages(filename)

    # get_fignums Return list of existing
    # figure numbers
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]

    # iterating over the numbers in list
    for fig in figs:
        # and saving the files
        fig.savefig(p, format='pdf')

        # close the object
    p.close()


def run_async(function, args_list, njobs, sleep_time_s=0.05):
    with Pool(njobs) as pool:
        with tqdm(total=len(args_list)) as pbar:
            results = {i: pool.apply_async(function, args=args)
                       for i, args in enumerate(args_list)}
            previous_state = 0
            while not all(future.ready() for future in results.values()):
                this_state = np.sum([future.ready() for future in results.values()])
                pbar.update(this_state - previous_state)
                previous_state = this_state
                sleep(sleep_time_s)
            results = [results[i].get() for i in range(len(results))]
    return results
