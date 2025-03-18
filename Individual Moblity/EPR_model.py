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
    def __init__(self):
        self.loc_label=defaultdict()





class EPR_model():
    def __init__(self, user_list, rho,gamma, beta_r, beta_t):
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

        self.rho=rho
        self.gamma=gamma

        self.resolution=10

    def simulation(self,usr_Status):


        output_mat = []
        for cnt, usr in enumerate(self.user_list):
            for step in range( usr_Status[usr].step_num):

                #####come to the time when the user need to make a decision
                temp = np.random.rand()
                P_new = self.rho * math.pow(len(usr_Status[usr].S_loc), self.gamma)

                ###Explore
                if temp < P_new:
                    keyword = 'explore'
                    current_label,current_loc,new_label,new_loc,move_r,move_a,stay_t,d_home= self.Explore_location(usr,usr_Status)
                else:
                    keyword = 'return'
                    current_label,current_loc,new_label,new_loc,move_r,move_a,stay_t,d_home= self.Return_location(usr,usr_Status)

                output_mat.append([usr, usr_Status[usr].home_label, step, keyword, current_label, new_label, current_loc[0], current_loc[1], new_loc[0], new_loc[1], move_r, move_a, stay_t, usr_Status[usr].current_t, d_home])

        #####save trajectory ##############
        output_mat = np.array(output_mat, dtype=object)
        columns = ['id', 'home_label', 'step', 'keyword', 'from_label', 'to_label', 'from_lat', 'from_lon',
                   'to_lat', 'to_lon', 'travel_d(km)', 'travel_angle', 'stay_t(h)', 'start', 'd_home']

        df = pd.DataFrame(data=output_mat, columns=columns)
        return df

    def Explore_location(self,usr,usr_each):
        '''
        :param usr: user_id
        :param usr_each: user_each_status
        :return: next move location
        '''

        current_loc = usr_each[usr].current_loc
        current_label = usr_each[usr].current_loc_label
        move_r = utils.GR_powerlaws(self.beta_r,0.03, 4000,1)[0]
        move_a =np.random.uniform(-180, 180, 1)[0]
        stay_t = utils.GR_powerlaws(self.beta_t,10, 24 * 60,1)[0]

        ####begin find new location
        new_loc = utils.find_move_loc(current_loc, move_r, move_a)
        new_label = utils.cells_to_h3(new_loc,self.resolution)

        if new_loc not in usr_each[usr].S_loc.keys():
            usr_each[usr].S_loc[new_loc] = 1
        else:
            usr_each[usr].S_loc[new_loc] = +1

        if new_label not in usr_each[usr].S_label.keys():
            usr_each[usr].S_label[new_label] = 1
        else:
            usr_each[usr].S_label[new_label] += 1


        usr_each[usr].current_loc = new_loc
        usr_each[usr].current_loc_label = new_label
        usr_each[usr].current_t += stay_t

        move_r =utils.haversine(current_loc, new_loc)
        d_home = utils.haversine(usr_each[usr].home, new_loc)
        move_a = None

        return current_label,current_loc,new_label,new_loc,move_r,move_a,stay_t,d_home


    def Return_location(self,usr,usr_each):
        '''
        :param usr: user_id
        :param usr_each: user_each_status
        :return: next move location
        '''
        current_loc = usr_each[usr].current_loc
        current_label = usr_each[usr].current_loc_label

        ####return
        temp_list = list(usr_each[usr].S_loc.values())
        prob = np.array(temp_list) / np.sum(temp_list)
        index = np.random.choice(range(len(prob)), p=prob)

        new_loc = list(usr_each[usr].S_loc.keys())[index]
        new_label = utils.cells_to_h3(new_loc,self.resolution)
        stay_t = utils.GR_powerlaws(self.beta_t,10, 24 * 60,1)[0]

        usr_each[usr].S_loc[new_loc] += 1
        usr_each[usr].S_label[new_label] += 1
        usr_each[usr].current_loc = new_loc
        usr_each[usr].current_loc_label = new_label
        usr_each[usr].current_t += stay_t

        move_r = utils.haversine(current_loc, new_loc)
        move_a = utils.get_bearing(current_loc, new_loc)
        d_home = utils.haversine(usr_each[usr].home, new_loc)

        return current_label,current_loc,new_label,new_loc,move_r,move_a,stay_t,d_home


def initialize(user_list, home_list, home_label_list, user_step_list):
    '''
    :param user_list: user_list
    :param home_list: user_home_list
    :param home_label_list: user_home_label_list
    :param user_step_list: user_step_list
    :return: dict of user status
    '''

    usr_Status = {}
    for usr, home, home_label,num_steps in zip(user_list, home_list, home_label_list,user_step_list):
        usr_Status[usr] = UserInfo(usr, home, home_label, num_steps)
        usr_Status[usr].S_loc[home] = 1
        usr_Status[usr].S_label[home_label] = 1

    return usr_Status



