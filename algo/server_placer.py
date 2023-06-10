import logging
from datetime import datetime
from typing import Iterable, List

import numpy as np

from data.base_station import BaseStation
from data.edge_server import EdgeServer
from utils import NO_LINK, DataUtils


class ServerPlacer(object):

    def __init__(self, base_stations: List[BaseStation],
                 distances: List[List[float]]):
        self.base_stations = base_stations.copy()
        self.edge_servers = None
        self.distances = distances

    def place_server(self, base_station_num, edge_server_num):
        raise NotImplementedError

    def _distance_edge_server_base_station(self, edge_server: EdgeServer,
                                           base_station: BaseStation) -> float:
        """
        Calculate distance between given edge server and base station
        
        :param edge_server: 
        :param base_station: 
        :return: distance(km)
        """

        def get_base_station_id(latitude, longitude):
            for base_station in self.base_stations:
                if abs(base_station.latitude -
                       latitude) < 1e-5 and abs(base_station.longitude -
                                                longitude) < 1e-5:
                    return base_station.id
            return None

        if not edge_server.base_station_id:
            edge_server.base_station_id = get_base_station_id(
                edge_server.latitude, edge_server.longitude)
        return self.distances[edge_server.base_station_id][base_station.id]

    def compute_objectives(self):
        objectives = {
            'latency': self.objective_latency(),
            'workload': self.objective_workload()
        }
        return objectives

    def objective_latency(self):
        """
        Calculate average edge server access delay (Average distance(km))
        """
        assert self.edge_servers
        total_delay = 0
        base_station_num = 0
        for es in self.edge_servers:
            for bs in es.assigned_base_stations:
                delay = self._distance_edge_server_base_station(es, bs)
                # if NO_LINK == delay:
                #     # 则该基站划入可达且负载最小服务器
                #     arriveable_es = []
                #     for es in self.edge_servers:
                #         dist = self._distance_edge_server_base_station(es, bs)
                #         if dist != NO_LINK:
                #             arriveable_es.append(es)
                #     min_es = None
                #     min_load = float('inf')
                #     for es in arriveable_es:
                #         if es.workload < min_load:
                #             min_load = es.workload
                #             min_es = es
                #             min_dist = self._distance_edge_server_base_station(
                #                 es, bs)
                #     if min_es:
                #         delay = min_dist
                #         min_es.workload += bs.workload
                #         es.workload -= bs.workload
                total_delay += delay
                base_station_num += 1

        print(base_station_num)
        # 保留2位小数
        return round(total_delay / base_station_num, 2)

    def objective_workload(self):
        """
        Calculate average edge server workload (Load standard deviation)
        
        Max worklaod of edge server - Min workload
        """
        assert self.edge_servers
        workloads = [e.workload for e in self.edge_servers]
        logging.debug("standard deviation of workload" + str(workloads))
        res = np.std(workloads)
        return round(res, 2)
