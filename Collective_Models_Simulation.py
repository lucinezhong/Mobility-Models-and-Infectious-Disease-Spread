import sys
import os
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))

relative_path='../Collective Mobility/'
sys.path.append(os.path.join(current_dir, relative_path))

import Collective_Models
print(current_dir)

def data_load():
    ####add real-world flow data
    df_flows = pd.read_csv('Dataset/sample_flow_data.csv')

    df_flows=df_flows[df_flows['from_label']!=df_flows['to_label']]
    df_flows=df_flows[df_flows['flow']>1]

    ####add frequency
    df_frequency = pd.read_csv('Dataset/max_min_f.csv')
    df_frequency['from_label'] = df_frequency['home_label']
    df_frequency['to_label'] = df_frequency['label']

    df_flows = df_flows.merge(df_frequency, how='left', on=['from_label', 'to_label'])
    df_flows['max_f'] = list(map(lambda x: 1 if x == None or np.isnan(x) == True else x, df_flows['max_f']))
    df_flows['min_f'] = list(map(lambda x: 1 if x == None or np.isnan(x) == True else x, df_flows['min_f']))


    return df_flows


if __name__ == "__main__":
    df_flows=data_load()

    Simulation = Collective_Models.Collective_Model(df_flows)


    df_flow_syn = Simulation.Estimate('Visitation_model')
    df_flow_syn.to_csv('results/flow_syn_Visitation.csv')

    df_flow_syn = Simulation.Estimate('Gravity_model')
    df_flow_syn.to_csv('results/flow_syn_Gravity.csv')

    #df_flow_syn = Simulation.Estimate('Radiation_model')
    #df_flow_syn.to_csv('results/flow_syn_Radiation.csv')



