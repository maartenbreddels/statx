import statx
import vaex
import numpy as np
import concurrent.futures
import multiprocessing
import threading
import timeit
import math


thread_count = multiprocessing.cpu_count()
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=thread_count)
print("threads", thread_count)


ds = vaex.open("gaia-dr1-10percent.hdf5")
x = ds.data.ra
binby = 'ra'

y = ds.data.dec
expr = 'dec'

binby2 = ['ra', 'dec']

limits = [0, 360]
limits2 = [[0, 360], [-180, 180]]
shape = 12
decimal = 1
chuck_size = 1024*1024
N_chunks = math.ceil(len(x) / chuck_size)
def chop(ar):
    return [ar[i * chuck_size:min(len(x), (i + 1) * chuck_size)] for i in range(N_chunks)]
chunks_x = chop(x)
chunks_y = chop(y)

print(len(x), np.sum(len(k) for k in chunks_x))
np.testing.assert_almost_equal(np.sum(x), np.sum(np.sum(k) for k in chunks_x), decimal=decimal)


def vaex_1():
    return ds.count(binby=binby, limits=limits, shape=shape)

def statx_1():
    grids = np.zeros((thread_count, shape), dtype=np.float64)
    lock = threading.Lock()
    thread_indices = iter(range(100))
    local = threading.local()
    def do_work(chunk_index):
        with lock:
            if not hasattr(local, 'index'):
                local.index = next(thread_indices)
        statx._count(grids[local.index], chunks_x[chunk_index], *limits)
    # map
    __ = list(thread_pool.map(do_work, range(len(chunks_x))))
    # reduce
    res = np.sum(grids, axis=0)
    return res


def vaex_2():
    return ds.sum(expr, binby=binby, limits=limits, shape=shape)

def statx_2():
    grids = np.zeros((thread_count, shape), dtype=np.float64)
    lock = threading.Lock()
    thread_indices = iter(range(100))
    local = threading.local()
    def do_work(chunk_index):
        with lock:
            if not hasattr(local, 'index'):
                local.index = next(thread_indices)
        statx._sum(grids[local.index], chunks_y[chunk_index], chunks_x[chunk_index], *limits)
    # map
    __ = list(thread_pool.map(do_work, range(len(chunks_x))))
    # reduce
    res = np.sum(grids, axis=0)
    return res



def vaex_3():
    return ds.count(binby=binby2, limits=limits2, shape=shape)

def statx_3():
    grids = np.zeros((thread_count, shape, shape), dtype=np.float64)
    lock = threading.Lock()
    thread_indices = iter(range(100))
    local = threading.local()
    def do_work(chunk_index):
        with lock:
            if not hasattr(local, 'index'):
                local.index = next(thread_indices)
        statx._count2(grids[local.index], chunks_x[chunk_index], *limits2[0], chunks_y[chunk_index], *limits2[1])
    # map
    __ = list(thread_pool.map(do_work, range(len(chunks_x))))
    # reduce
    res = np.sum(grids, axis=0)
    return res


if __name__ == "__main__":
    res_vaex_3 = vaex_3()
    res_statx_3 = statx_3()
    np.testing.assert_almost_equal(res_vaex_3, res_statx_3, decimal=decimal)



    res_vaex_2 = vaex_2()
    res_statx_2 = statx_2()
    np.testing.assert_almost_equal(res_vaex_2, res_statx_2, decimal=decimal)

    res_vaex_1 = vaex_1()
    res_statx_1 = statx_1()
    np.testing.assert_almost_equal(res_vaex_1, res_statx_1, decimal=decimal)


    N = 3
    print("job 1 (count2d)")
    t_total = timeit.timeit('statx_3()', setup='from __main__ import statx_3', number=N)
    t = t_total / N
    print("spectabular:\t%5.3f seconds" % t)


    t_total = timeit.timeit('vaex_3()',  setup='from __main__ import vaex_3', number=N)
    t = t_total / N
    print("vaex:\t\t%5.3f seconds" % t)


    print("job 2 (count)")
    t_total = timeit.timeit('statx_1()', setup='from __main__ import statx_1', number=N)
    t = t_total / N
    print("spectabular:\t%5.3f seconds" % t)


    t_total = timeit.timeit('vaex_1()',  setup='from __main__ import vaex_1', number=N)
    t = t_total / N
    print("vaex:\t\t%5.3f seconds" % t)


    print("job 3 (sum)")
    t_total = timeit.timeit('statx_2()', setup='from __main__ import statx_2', number=N)
    t = t_total / N
    print("spectabular:\t%5.3f seconds" % t)


    t_total = timeit.timeit('vaex_2()',  setup='from __main__ import vaex_2', number=N)
    t = t_total / N
    print("vaex:\t\t%5.3f seconds" % t)


