import logging
from datetime import datetime
from typing import Iterable, List

# 导入所需的库
import numpy as np
import pandas as pd
from data.edge_server import EdgeServer
from sklearn.cluster import DBSCAN, KMeans
from sklearn_extra.cluster import KMedoids

from .server_placer import ServerPlacer

# class EdgeServer(object):

#     def __init__(self, id, latitude, longitude, base_station_id=None):
#         self.id = id
#         self.latitude = latitude
#         self.longitude = longitude
#         self.base_station_id = base_station_id
#         self.assigned_base_stations = []
#         self.workload = 0


class SDTSServerPlacer(ServerPlacer):
    """
    SDTS approach
    """
    name = 'SDTS'

    def place_server(self, base_station_num, edge_server_num):
        logging.info("{0}:Start running SDTS with N={1}, K={2}".format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), base_station_num,
            edge_server_num))
        # init data as ndarray
        base_stations = self.base_stations[:base_station_num]
        coordinates = list(
            map(lambda x: (x.latitude, x.longitude), base_stations))
        data = np.array(coordinates)
        print("data size", len(data))
        # 创建DBSCAN对象，设置参数eps和min_samples
        eps, min_samples = 0.008, 6

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)

        label = dbscan.fit_predict(data)
        # 簇的个数
        print("clust_cnt", len(set(label)))
        # 获取所有标签的噪声的列表
        Noise = data[label == -1]
        # 对R中的数据按照标签进行分类
        R = [data[label == i] for i in range(len(set(label)))]
        # 有效点数量
        valid_bs_cnt = sum([len(Ri) for Ri in R])
        apha = edge_server_num / valid_bs_cnt
        # 噪声点数量
        print("Noise Cnt", len(Noise))
        R_new = []
        for Ri in R:
            bs_cnt = len(Ri)
            svr_cnt = bs_cnt * apha

            if svr_cnt < 1:
                Noise = np.concatenate((Noise, Ri), axis=0)
            else:
                R_new.append(Ri)
        R = R_new
        print("R_new Cnt", len(R_new))
        # 将所有噪声加入离它最近的簇中
        for noise in Noise:
            min_di = 100000
            min_index = 0
            for i in range(len(R)):
                Ri = R[i]
                for bs in Ri:
                    di = np.sqrt(np.sum(np.square(noise - bs)))
                    if di < min_di:
                        min_di = di
                        min_index = i
            R[min_index] = np.concatenate((R[min_index], [noise]), axis=0)
        # 有效点数量
        print("point Cnt", sum([len(Ri) for Ri in R]))

        def get_base_station_id(latitude, longitude):
            for base_station in base_stations:
                if abs(base_station.latitude -
                       latitude) < 0.0001 and abs(base_station.longitude -
                                                 longitude) < 0.0001:
                    return base_station.id
            return None

        edge_servers = []
        scheduler_servers = []
        used_svr_cnt = 0
        for Ri in R:
            bs_cnt = len(Ri)
            svr_cnt = bs_cnt * apha
            print(bs_cnt, apha, svr_cnt)
        # 对R按照元素数量排序
        R.sort(key=lambda x: -len(x))
        for Ri in R:
            svr_cnt = len(Ri) * apha
            if used_svr_cnt >= edge_server_num:
                break
            if svr_cnt < 1:
                continue
            # KMedoides聚类
            kmedoids = KMedoids(n_clusters=round(svr_cnt),
                                random_state=0).fit(Ri)
            # 获取聚类中心
            centers = kmedoids.cluster_centers_
            # 获取每个簇对应的基站列表
            km_label = kmedoids.labels_
            bs_clus_list = []
            for i in range(len(centers)):
                bs_clus_list.append([])
            for i in range(len(km_label)):
                bs_clus_list[km_label[i]].append(Ri[i])
            sub_edge_servers = []
            for clus,center in enumerate(centers):
                used_svr_cnt += 1
                if used_svr_cnt >= edge_server_num:
                    break
                base_station_id=get_base_station_id(
                                             center[0], center[1])
                edge_server = EdgeServer(id=used_svr_cnt,
                                         latitude=center[0],
                                         longitude=center[1],
                                         base_station_id=base_station_id)

                # 设置该边缘服务器负责的基站
                edge_server.assigned_base_stations = []
                for bs in bs_clus_list[clus]:
                    base_station_id = get_base_station_id(bs[0], bs[1])
                    base_station = self.base_stations[base_station_id]
                    edge_server.workload += base_station.workload
                    edge_server.assigned_base_stations.append(base_station)

                sub_edge_servers.append(edge_server)
            if not len(sub_edge_servers):
                continue
            # cenrters的距离中心点最近的点作为调度服务器
            min_index = 0
            min_di = 100000000
            for i in range(len(centers[:len(sub_edge_servers)])):
                di = np.sqrt(np.sum(np.square(centers[i] - Ri.mean(axis=0))))
                if di < min_di:
                    min_di = di
                    min_index = i
            # print(min_index)
            scheduler_servers.append(sub_edge_servers[min_index])
            edge_servers += sub_edge_servers
                
        self.edge_servers = edge_servers
            
        print(
            f"used_svr_cnt:{used_svr_cnt}, edge_server_num:{edge_server_num}")
        logging.info("{0}:End running SDTS".format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
