import numba
import progressbar
import multiprocessing.pool
import numpy as np

def execute_kernel2D(kernel, width, height):
    image = np.ones((height, width), np.float32)

    @numba.njit(nogil=True)
    def inner_loop(image, y):
        for x in range(width):
            image[y, x] = kernel(x, y)
        return y

    bar = progressbar.ProgressBar(max_value=height)
    with multiprocessing.pool.ThreadPool() as pool:
        # Manual multi-threading so we can have our nice progressbar
        for y in pool.imap(lambda y: inner_loop(image, y), range(height)):
            bar.update(y)
    return image


def transform_reduce_2D_to_1D(kernel, width, height):
    image = np.ones((height, width), np.float32)

    @numba.njit(nogil=True)
    def inner_loop(image, y):
        result = []
        for x in range(width):
            r = kernel(x, y)
            if r is not None:
                result.append(r)
        return result

    results_list = []
    bar = progressbar.ProgressBar(max_value=height)
    with multiprocessing.pool.ThreadPool() as pool:
        # Manual multi-threading so we can have our nice progressbar
        for y, partial_list in enumerate(pool.imap(lambda y: inner_loop(image, y), range(height))):
            results_list.append(partial_list)
            bar.update(y)

    # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    return [item for sublist in results_list for item in sublist]
