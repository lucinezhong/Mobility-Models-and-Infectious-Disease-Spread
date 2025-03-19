import copy
import math
import networkx as nx
import numpy as np
import pandas


def normalized_net(net):
    H = copy.deepcopy(net)
    for i in H.nodes():
        sumvalue = sum([net[i][x]['weight'] for x in net.neighbors(i)])
        for j in H.neighbors(i):
            if sumvalue != 0:
                H[i][j]['weight'] = (net[i][j]['weight'] + 1) / sumvalue
                H[i][j]['distance'] = (1 - math.log(H[i][j]['weight']))  ## for directed netowrk, make caution
    return H


def Effective_distance(net):
    Eff_D = dict(nx.all_pairs_dijkstra_path_length(net, weight='distance'))  ########effective distance
    return Eff_D


def Effective_distance_mutiple_OL(Eff_D,OL_list,node_list):
    """
    :param distanceNo_dict: shortest_path distance
    :param distance_dict: effective distance
    :param sourcelist: source
    :return: country distancing
    """
    M=len(node_list)

    new_Eff_D = dict(zip(node_list, [0 for i in np.arange(len(node_list))]))
    for OL in OL_list:
        if OL in Eff_D.keys():
            for key, valuesource in Eff_D[OL].items():
                new_Eff_D[key] += 1/math.exp(Eff_D[OL][key])

    for key, value in new_Eff_D.items():
        if value == 0:
            new_Eff_D[key] = 0
        else:
            new_Eff_D[key] = math.log(1/new_Eff_D[key] )
    return new_Eff_D
