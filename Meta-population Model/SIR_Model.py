import pandas as pd
import numpy as np

class NodeInfo:
    def __init__(self, infectnum, recovernum, susceptnum, infectnumold, recovernumold, susceptnumold,population,arrivalT):
        self.infectnum = float(infectnum)
        self.recovernum = float(recovernum)
        self.susceptnum = float(susceptnum)
        self.infectnumold = float(infectnumold)
        self.recovernumold = float(recovernumold)
        self.susceptnumold = float(susceptnumold)

        self.population = int(population)
        self.arrivalT=arrivalT


def sigmoidfun(x, ebu):
    x = x / ebu
    sigma = pow(x, 4) / (1 + pow(x, 4))
    return float(sigma)

# epidemic model SIR
def SIRmetapopulationModel(net, steps, time_interval,para,OL_LIST, OL_infections):
    '''
    :param net: flow network
    :param steps: simulation stpes
    :param time_interval: time_interval
    :param para: infectionrate, reoveryrate, migrationrate,ebsong
    :param OL_LIST: outbreak location list
    :param OL_infections: initial outbreak location infections
    :return:
    '''

    [infectionrate, reoveryrate, migrationrate,ebsong]=para
    output_mat=[]

    NodeStatus = {}
    for i in net.nodes():
        NodeStatus[i] = NodeInfo(0, 0, 1, 0, 0, 1,net.nodes[i]['population'],-1)
    for i,x in zip(OL_LIST,OL_infections):
        if i in net.nodes():
            NodeStatus[i].infectnum = x/ NodeStatus[i].population
            NodeStatus[i].susceptnum= NodeStatus[i].susceptnum-(x / NodeStatus[i].population)
            output_mat.append([0,i,NodeStatus[i].population,NodeStatus[i].infectnum,NodeStatus[i].susceptnum])

    for t in range(1, steps):
        for i in net.nodes():
            NodeStatus[i].infectnumold = NodeStatus[i].infectnum
            NodeStatus[i].susceptnumold = NodeStatus[i].susceptnum
            NodeStatus[i].recovernumpld = NodeStatus[i].recovernum

        for i in net.nodes():

            ####self infections and recovery

            newinfectionrate = infectionrate* sigmoidfun(NodeStatus[i].infectnumold, ebsong)

            print(t,NodeStatus[i].infectnumold) #,sigmoidfun(NodeStatus[i].infectnumold, ebsong))
            tempinfect = newinfectionrate * NodeStatus[i].susceptnumold * NodeStatus[i].infectnumold - reoveryrate * NodeStatus[i].infectnumold
            tempsuscept = -newinfectionrate * NodeStatus[i].susceptnumold * NodeStatus[i].infectnumold
            temprecover = reoveryrate * NodeStatus[i].infectnumold

            for j in net.neighbors(i): ###out-flow
                weight = migrationrate * net[i][j]['weight']
                if NodeStatus[j].infectnumold - NodeStatus[i].infectnumold < 0:
                    tempinfect = tempinfect + weight * (NodeStatus[j].infectnumold - NodeStatus[i].infectnumold)
                if NodeStatus[j].susceptnumold - NodeStatus[i].susceptnumold < 0:
                    tempsuscept = tempsuscept + weight * (NodeStatus[j].susceptnumold - NodeStatus[i].susceptnumold)
                if NodeStatus[j].recovernumold - NodeStatus[i].recovernumold < 0:
                    temprecover = temprecover + weight * (NodeStatus[j].recovernumold - NodeStatus[i].recovernumold)

            for j in net.predecessors(i): ###in-flow
                weight = migrationrate * net[j][i]['weight']
                if  NodeStatus[j].infectnumold - NodeStatus[i].infectnumold>0:
                    tempinfect = tempinfect + weight * (NodeStatus[j].infectnumold - NodeStatus[i].infectnumold)
                if NodeStatus[j].susceptnumold - NodeStatus[i].susceptnumold > 0:
                    tempsuscept = tempsuscept + weight * (NodeStatus[j].susceptnumold - NodeStatus[i].susceptnumold)
                if  NodeStatus[j].recovernumold - NodeStatus[i].recovernumold>0:
                    temprecover = temprecover + weight * (NodeStatus[j].recovernumold - NodeStatus[i].recovernumold)

            NodeStatus[i].infectnum = NodeStatus[i].infectnumold + (tempinfect) * time_interval
            NodeStatus[i].susceptnum = NodeStatus[i].susceptnumold + (tempsuscept) * time_interval
            NodeStatus[i].recovernum = NodeStatus[i].recovernumold + (temprecover) * time_interval

            output_mat.append([t, i, NodeStatus[i].population, NodeStatus[i].susceptnum , NodeStatus[i].infectnum])

    df=pd.DataFrame(np.mat(output_mat),columns=['time','node','population','susceptnum','infectnum'])
    return df


