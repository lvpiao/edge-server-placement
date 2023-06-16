import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.utils import resample
import multiprocessing
from algo.kmeans import *
from algo.mlp import *
from algo.random import *
from algo.sdts import *
from algo.topk import *
from algo.weighted_kmeans import *
from utils import *


def run_with_settings(placer, n, k, repeat_times=1):
    if repeat_times == 1:
        placer.place_server(n, k)
        objectives = placer.compute_objectives()
    else:
        # run multiple times to obtain the mean value
        objectives_list = []
        for t in range(repeat_times):
            placer.place_server(n, k)
            one_objectives = placer.compute_objectives()
            time.sleep(1)
            objectives_list.append(one_objectives)

        objectives = {}
        for k in objectives_list[-1].keys():
            mean_value = sum(o[k]
                             for o in objectives_list) / len(objectives_list)
            objectives[k] = round(mean_value, 2)
    return objectives


def task(args):
    # unpack the arguments
    name, placer, n, k = args
    # run the function with the settings
    settings = {
        'num_base_stations': n,
        'num_edge_servers': k,
        'placer_name': name
    }
    objectives = run_with_settings(placer, n, k, 1)
    record = {**settings, **objectives}
    return record


import copy


def run(placers, results_fpath='results/results.csv'):
    # create a list of arguments for each task using list comprehension
    args_list = [(name, copy.deepcopy(placer), 3000, k)
                 for k in range(100, 600, 100)
                 for name, placer in placers.items()]

    # cpu cores
    num_processes = multiprocessing.cpu_count()
    # create a pool of processes
    pool = multiprocessing.Pool(num_processes - 2)

    # map the tasks to the pool and get the results
    records = pool.map(task, args_list)

    # close the pool
    pool.close()

    pd_records = pd.DataFrame(records)
    # 根据placer_name和num_edge_servers排序
    pd_records = pd_records.sort_values(by=['placer_name', 'num_edge_servers']).T
    pd_records.to_csv(results_fpath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data = DataUtils('./dataset/base_stations_min.csv',
                     './dataset/data_min.csv')
    placers = {}
    # placers['MIP'] = MIPServerPlacer(data.base_stations, data.distances)
    placers['K-means'] = KMeansServerPlacer(data.base_stations, data.distances)
    placers['Top-K'] = TopKServerPlacer(data.base_stations, data.distances)
    placers['Random'] = RandomServerPlacer(data.base_stations, data.distances)
    # placers['weighted_k_means'] = WeightedKMeansServerPlacer(data.base_stations, data.distances)
    placers['SDTS'] = SDTSServerPlacer(data.base_stations, data.distances)
    run(placers)