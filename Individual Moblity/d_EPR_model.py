
import sys
import os
from collections import defaultdict

current_dir = os.path.dirname(os.path.realpath(__file__))

relative_path='../Individual Mobility/'
sys.path.append(os.path.join(current_dir, relative_path))

import numpy as np
import pandas as pd
import math
import h3
import utils
import warnings
warnings.filterwarnings("ignore")



class UserInfo:
    def __init__(self, id, home_lat_lon,home_label,step_num):
        self.id = id
        self.home =(round(home_lat_lon[0],3),round(home_lat_lon[1],3))
        self.home_label =home_label

        self.current_loc=self.home
        self.current_loc_label= self.home_label

        self.current_t = 0
        self.step_num = step_num
        self.step_time = 0

        ####saved location and visit frequency
        self.new_loc = -1
        self.new_loc_county=-1

        self.S_loc = dict()  ######visited locations and their frequency
        self.S_label=dict()


class region_grid:
    def __init__(self,loc_loc_prob,loc_loc_distance,county_pos):
        self.loc_loc_prob = loc_loc_prob
        self.loc_loc_distance = loc_loc_distance
        self.county_pos = county_pos
        self.loc_county=dict()

class d_EPR_model():
    def __init__(self, user_list, rho, gamma, beta_r, beta_t):
        """
        :param user_list: user_list
        :param rho: explore para
        :param gamma: exponent for visited locations
        :param beta_t: exponent for stay duration
        :param beta_r: exponent for distance
        :return: individual trajectory dataframe
        """

        self.user_list = user_list
        self.verbose = False

        self.beta_r = beta_r
        self.beta_t = beta_t

        self.rho = rho
        self.gamma = gamma

        self.resolution = 10

    def simulation(self,usr_Status,region_Status):

        output_mat = []
        for cnt, usr in enumerate(self.user_list):
            for step in range(0, usr_Status[usr].step_num):

                #####come to the time when the user need to make a decision
                temp = np.random.rand()
                P_new = self.rho * math.pow(len(usr_Status[usr].S_loc), self.gamma)

                ###Explore
                if temp < P_new:
                    keyword = 'explore'
                    current_label,current_loc,new_label,new_loc,move_r,move_a,stay_t,d_home= self.Explore_location(usr,usr_Status,region_Status)
                else:
                    keyword = 'return'
                    current_label,current_loc,new_label,new_loc,move_r,move_a,stay_t,d_home= self.Return_location(usr,usr_Status, region_Status)

                output_mat.append([usr, usr_Status[usr].home_label, step, keyword, current_label, new_label, current_loc[0], current_loc[1], new_loc[0], new_loc[1], move_r, move_a, stay_t, usr_Status[usr].current_t, d_home])

        #####save trajectory ##############
        output_mat = np.array(output_mat, dtype=object)
        columns = ['id', 'home_label', 'step', 'keyword', 'from_label', 'to_label', 'from_lat', 'from_lon',
                   'to_lat', 'to_lon', 'travel_d(km)', 'travel_angle', 'stay_t(h)', 'start', 'd_home']

        df = pd.DataFrame(data=output_mat, columns=columns)
        return df

    def Explore_location(self,usr,usr_each,region_Status):
        '''
        :param usr: user_id
        :param usr_each: user_each_status
        :param  region_Status: region_Status
        :return: next move location
        '''

        current_loc = usr_each[usr].current_loc
        current_label= usr_each[usr].current_loc_label
        stay_t = utils.GR_powerlaws(self.beta_t,10, 24 * 60,1)[0]

        temp_neighbour = list(region_Status.loc_loc_prob[current_label].keys())
        temp_prob = list(region_Status.loc_loc_prob[current_label].values())

        while 1:
            prob = np.array(temp_prob) / np.sum(temp_prob)
            index = np.random.choice(range(len(prob)), p=prob)
            new_label = temp_neighbour[index]
            if new_label!=current_label and new_label in region_Status.county_pos.keys():
                break

        new_loc = region_Status.county_pos[new_label]

        if new_loc not in usr_each[usr].S_loc.keys():
            usr_each[usr].S_loc[new_loc] = 1
        else:
            usr_each[usr].S_loc[new_loc] = +1

        if new_label not in usr_each[usr].S_label.keys():
            usr_each[usr].S_label[new_label] = 1
        else:
            usr_each[usr].S_label[new_label] += 1

        usr_each[usr].current_loc = new_loc
        usr_each[usr].current_loc_county = new_label
        usr_each[usr].current_t += stay_t

        move_r =utils.haversine(current_loc, new_loc)
        d_home = utils.haversine(usr_each[usr].home, new_loc)
        move_a = None

        return current_label,current_loc,new_label,new_loc,move_r,move_a,stay_t,d_home



    def Return_location(self,usr,usr_each,region_Status):
        '''
        :param usr: user_id
        :param usr_each: user_each_status
        :param  region_Status: region_Status
        :return: next move location
        '''
        current_loc = usr_each[usr].current_loc
        current_label= usr_each[usr].current_loc_label
        stay_t = utils.GR_powerlaws(self.beta_t, 10, 24 * 60, 1)[0]

        temp_list = list(usr_each[usr].S_label.values())
        prob = np.array(temp_list) / np.sum(temp_list)
        index = np.random.choice(range(len(prob)), p=prob)
        new_label= list(usr_each[usr].S_label.keys())[index]
        new_loc = region_Status.county_pos[new_label]

        usr_each[usr].S_loc[new_loc] += 1
        usr_each[usr].S_label[new_label] += 1

        usr_each[usr].current_loc = new_loc
        usr_each[usr].current_loc_label= new_label
        usr_each[usr].current_t += stay_t * 2

        move_r = utils.haversine(current_loc, new_loc)
        move_a = utils.get_bearing(current_loc, new_loc)
        d_home = utils.haversine(usr_each[usr].home, new_loc)

        return current_label,current_loc,new_label,new_loc,move_r,move_a,stay_t,d_home


def initialize(user_list, home_list,home_label_list,user_step_list, loc_loc_prob,loc_loc_distance,county_pos):
    region_Status=region_grid(loc_loc_prob,loc_loc_distance,county_pos)

    usr_Status = {}
    for usr,home,home_label,num_steps in zip(user_list,home_list,home_label_list,user_step_list):
        ####reseting home to countty
        home_label=utils.cells_to_county(home)
        home= region_Status.county_pos[home_label]

        usr_Status[usr] = UserInfo(usr, home,home_label,num_steps)
        usr_Status[usr].S_loc[home] = 1
        usr_Status[usr].S_label[home_label] = 1

        region_Status.loc_county[home]=home_label

    return usr_Status,region_Status
