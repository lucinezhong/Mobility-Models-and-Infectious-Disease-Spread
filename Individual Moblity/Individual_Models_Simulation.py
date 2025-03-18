import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))

relative_path='../Individual Mobility/'
sys.path.append(os.path.join(current_dir, relative_path))

import pandas as pd
import numpy as np
import pickle

import EPR_model
import d_EPR_model

def simulation_setting():
    user_list=[0,1,2,3,4]
    home_list=list(zip([42.3,42.3,42.3,42.3,42.3],[-71.0,-71.0,-71.0,-71.0,-71.0]))
    home_label=['8a2a3029a8b7fff','8a2a3029a8b7fff','8a2a3029a8b7fff','8a2a3029a8b7fff','8a2a3029a8b7fff']
    user_step_list=[100,150,200,150,100]
    return user_list,home_list,home_label,user_step_list

def county_flow_data_load():
    with open('Dataset/loc_loc_prob.pickle', 'rb') as handle:
        loc_loc_prob = pickle.load(handle)

    with open('Dataset/loc_loc_distance.pickle', 'rb') as handle:
        loc_loc_distance = pickle.load(handle)

    with open( 'Dataset/county_pos.pickle', 'rb') as handle:
        county_pos = pickle.load(handle)

    with open('Dataset/pos_county.pickle', 'rb') as handle:
        pos_county = pickle.load(handle)

    return loc_loc_prob, loc_loc_distance,county_pos,pos_county

def parameter_setting(model_which):
    if model_which=='EPR_model' or model_which=='d_EPR_model':
        rho=0.6
        gamma=-0.21
        beta_r=-1.2
        beta_t=-1.2

    return rho, gamma, beta_r, beta_t

if __name__ == "__main__":

    model_which = 'EPR_model'
    model_which = 'd_EPR_model'

    if model_which=='EPR_model':
        rho, gamma, beta_r, beta_t = parameter_setting(model_which)
        user_list, home_list, home_label, user_step_list=simulation_setting()

        ####start
        usr_Status = EPR_model.initialize(user_list, home_list, home_label, user_step_list)
        model_test = EPR_model.EPR_model(user_list, rho, gamma, beta_r, beta_t)
        df = model_test.simulation(usr_Status)

    if model_which=='d_EPR_model':
        rho, gamma, beta_r, beta_t = parameter_setting(model_which)
        user_list, home_list, home_label, user_step_list = simulation_setting()
        loc_loc_prob, loc_loc_distance, county_pos, pos_county = county_flow_data_load()

        usr_Status,region_Status = d_EPR_model.initialize(user_list, home_list, home_label, user_step_list,loc_loc_prob, loc_loc_distance, county_pos)
        model_test = d_EPR_model.d_EPR_model(user_list, rho, gamma, beta_r, beta_t)
        df = model_test.simulation(usr_Status, region_Status)



    df.to_csv('Results/'+model_which+'_individual_trajectory.csv')

