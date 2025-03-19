import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))

relative_path='../Meta-population Model/'
sys.path.append(os.path.join(current_dir, relative_path))

import networkx as nx
import pandas as pd
import numpy as np

import SIR_Model
import utils


def load_mobility_network():
    ####load moiblity network
    net = nx.read_edgelist('Dataset/GlobalCountry_airline.edgelist', create_using=nx.DiGraph())
    edge_list=[]
    for i in net.nodes():
        for j in nx.neighbors(net, i):
            if net[i][j]['weight'] <100:
                edge_list.append((i,j))
            if i==j:
                edge_list.append((i, j))
    net.remove_edges_from(edge_list)

    net=utils.normalized_net(net)

    ####load population
    Countrypop = pd.read_csv('Dataset/GlobalCountry_pop.csv')
    codePop = dict(zip(Countrypop['countryCode'], Countrypop['Population']))
    for i in net.nodes():
        if i in codePop.keys():
            net.nodes[i]['population'] = codePop[i]
        else:
            net.nodes[i]['population'] = 10000
    return net

def extract_arraival_time(df_simulation,Eff_D,Eff_D_mutiple):
    '''
    :param df_simulation: simulated infections
    :param Eff_D: Effective distance
    :param Eff_D_mutiple: Mutiple-OLs effective distance
    :return:
    '''
    df_simulation['infectnum']=df_simulation['infectnum'].astype(float)
    df_simulation['population'] = df_simulation['population'].astype(float)

    output_mat=[]

    for node,df_temp in df_simulation.groupby(['node']):
        df_temp=df_temp[df_temp['infectnum']*df_temp['population']>=1]
        arrival_T=df_temp['time'].min()
        output_mat.append([node,arrival_T,Eff_D[node],Eff_D_mutiple[node]])

    df=pd.DataFrame(np.mat(output_mat),columns=['node','arrival_T','Eff_D','Eff_D_multiple'])
    return df

if __name__ == "__main__":

    test_which='Single_OL'
    test_which = 'Multiple_OL'

    if test_which=='Single_OL':
        OL_list=['CN'] ##Outbreak Location (OL)
        OL_infections=[10] ##Outbreak Location (OL) intial infectiaons

    if test_which == 'Multiple_OL':

        OL_list = ['CN','IT','JP','AU']  ##Outbreak Location (OL)
        OL_infections = [10,10,10,10]  ##Outbreak Location (OL) intial infectiaons

    para=[0.35,0.035,0.02,0.00000000002  ] #infectionrate, reoveryrate, migrationrate,ebsong

    #### load network and computed distance
    net=load_mobility_network()
    Eff_D = utils.Effective_distance(net)
    Eff_D_multiple=utils.Effective_distance_mutiple_OL(Eff_D, OL_list, list(net.nodes()))

    #### simulating infectious disease spread
    df_simulation=SIR_Model.SIRmetapopulationModel(net, 1000, 0.1, para, OL_list, OL_infections)

    ### computing arrival time
    df=extract_arraival_time(df_simulation, Eff_D[OL_list[0]],Eff_D_multiple)

    df_simulation.to_csv('Results/Infection_simulation.csv')
    df.to_csv('Results/Effective_Distance_'+test_which+'.csv')
