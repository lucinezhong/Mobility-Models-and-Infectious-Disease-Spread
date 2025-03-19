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
    def __init__(self, id_num, home_lat_lon, home_label, step_num, rgc_exponent):
        self.id = id_num
        self.home = home_lat_lon
        self.home_label = home_label

        self.current_loc = self.home
        self.current_loc_h3 = utils.cells_to_h3(self.home,10)
        self.current_module = self.current_loc

        self.new_loc = -1
        self.new_loc_h3 = -1
        self.new_module = -1

        self.current_t = 0
        self.step_num = step_num
        self.step_time = 0

        self.S_loc = dict()  ######visited locations and their frequency

        #####save_explore_module_frequncy
        self.S_module_d_home = dict()######visited modules distance from home
        self.S_module_loc = dict() ######visited locations within module
        self.S_module = dict()  ######visited modules and their frequency
        self.S_module_radius = dict() ######visited modules radius

        #####users  rgc
        self.rgc = utils.GR_powerlaws(rgc_exponent, 1, 4000, 1)[0]


class Switch_model():
    def __init__(self, user_list, prob_switch, gamma_w, gamma_c_slope, rho_w, rho_c, beta_r, beta_t):
        """
        :param user_list: user_list
        :param prob_switch: prob_switch
        :param gamma_w, rho_w: parameters for within-module travels
        :param gamma_c_slope, rho_c: parameters for cross-module travels
        :param beta_t: exponent for stay duration
        :param beta_r: exponent for distance
        :return: individual trajectory dataframe
        """

        self.user_list = user_list
        self.verbose = False

        self.beta_r = beta_r
        self.beta_t = beta_t

        self.prob_switch = prob_switch
        self.rho_w = rho_w
        self.gamma_w = gamma_w
        self.rho_c = rho_c
        self.gamma_c_slope = gamma_c_slope
        self.fixed_module_size=10 ####fixed number of locations within module

        self.resolution=10

    def simulation(self, usr_Status):
        '''
        :param usr_Status:  usr_Status
        :return:
        '''
        output_mat = []
        for cnt, usr in enumerate(self.user_list):
            print(usr, usr_Status[usr].step_num)
            for step in range(0, usr_Status[usr].step_num):
                print(usr,usr_Status[usr].step_num)
                current_module = usr_Status[usr].current_module
                current_d_home = usr_Status[usr].S_module_d_home[current_module]

                S_w_loc = len(np.unique(usr_Status[usr].S_module_loc[current_module]))
                P_w = self.rho_w * math.pow(S_w_loc, self.gamma_w)


                S_c_loc = len(list(usr_Status[usr].S_module.keys()))
                new_gamma_c = -0.6 + math.log10(usr_Status[usr].rgc + 1) * self.gamma_c_slope
                P_c = self.rho_c * math.pow(S_c_loc, new_gamma_c)

                P_switch = self.prob_switch

                temp = np.random.rand()
                if temp < 1 - P_switch:
                    #####within-module travel
                    temp = np.random.rand()
                    if temp <= P_w and S_w_loc <= self.fixed_module_size:
                        keyword = 'within_explore'
                        current_h3, current_loc, new_h3, new_loc, move_r, move_a, stay_t, d_home = self.Explore(usr,
                                                                                                                usr_Status,
                                                                                                                keyword)

                    else:
                        keyword = 'within_return'
                        current_h3, current_loc, new_h3, new_loc, move_r, move_a, stay_t, d_home = self.Return(usr,
                                                                                                               usr_Status,
                                                                                                               keyword)

                else:
                    #####cross-module travel
                    temp = np.random.rand()
                    if temp < P_c:
                        keyword = 'cross_explore'
                        current_h3, current_loc, new_h3, new_loc, move_r, move_a, stay_t, d_home = self.Explore(usr,
                                                                                                                usr_Status,
                                                                                                                keyword)

                    else:
                        keyword = 'cross_return'
                        current_h3, current_loc, new_h3, new_loc, move_r, move_a, stay_t, d_home = self.Return(
                            usr, usr_Status, keyword)

                #####save trajectory ##############
                output_mat.append(
                    [usr, usr_Status[usr].home_label, step, keyword, current_h3, new_h3, current_loc[0], current_loc[1],
                     new_loc[0], new_loc[1], move_r, move_a, stay_t, usr_Status[usr].current_t, d_home])

        output_mat = np.array(output_mat, dtype=object)
        columns = ['id', 'home_label', 'step', 'keyword', 'from_label', 'to_label', 'from_lat', 'from_lon', 'to_lat',
                   'to_lon', 'travel_d(km)', 'travel_angle', 'stay_t(h)', 'start', 'd_home']

        df = pd.DataFrame(data=output_mat, columns=columns)
        return df

    def Explore(self, usr, usr_each, keyword):
        '''
        :param usr: user_id
        :param usr_each: user_each_status
        :param keyword: keyword
        :return: next move location
        '''

        current_loc = usr_each[usr].current_loc
        current_loc_h3 = usr_each[usr].current_loc_h3
        current_module = usr_each[usr].current_module
        current_d_home = usr_each[usr].S_module_d_home[current_module]


        move_a = utils.GR_random(-180, 180, 1)[0]
        stay_t = utils.GR_powerlaws(self.beta_t, 10, 24 * 60, 1)[0]


        if keyword == 'within_explore':
            ####begin find new location
            move_r = utils.GR_powerlaws(self.beta_r, 0.03, 20, 1)[0]
            new_loc = utils.find_move_loc(current_loc, move_r, move_a)  ####
            new_loc_h3 = utils.cells_to_h3(new_loc,self.resolution)

            new_module = current_module
            usr_each[usr].S_module_loc[new_module].append(new_loc)


        if keyword == 'cross_explore':
            ####begin find new module
            move_r =utils.GR_powerlaws(self.beta_r,20,4000,1)[0]
            new_loc = utils.find_move_loc(current_loc, move_r, move_a)
            new_loc_h3 = utils.cells_to_h3(new_loc,self.resolution)
            new_module = new_loc

            ###explore a new module
            usr_each[usr].S_module[new_loc] = 1  #####frequency
            usr_each[usr].S_module_loc[new_loc] = [new_loc]  #####locations sequence
            usr_each[usr].S_module_d_home[new_loc] = utils.haversine(usr_each[usr].home, new_loc)


        usr_each[usr].current_loc = new_loc
        usr_each[usr].current_loc_h3 = new_loc_h3
        usr_each[usr].current_module = new_module
        usr_each[usr].S_loc[new_loc] = 1
        usr_each[usr].current_t += stay_t
        d_home = usr_each[usr].S_module_d_home[new_module]

        return current_loc_h3, current_loc, new_loc, new_loc, move_r, move_a, stay_t, d_home

    def Return(self, usr, usr_each, keyword):
        '''
        :param usr: user_id
        :param usr_each: user_each_status
        :param keyword: keyword
        :return: next move location
        '''

        current_loc = usr_each[usr].current_loc
        current_loc_h3 = usr_each[usr].current_loc_h3
        current_module = usr_each[usr].current_module
        stay_t = utils.GR_powerlaws(self.beta_t, 10, 24 * 60, 1)[0]

        if keyword == 'within_return':
            within_loc = usr_each[usr].S_module_loc[current_module]

            temp_list = [usr_each[usr].S_loc[i] for i in within_loc]
            prob = np.array(temp_list) / np.sum(temp_list)
            index = np.random.choice(range(len(prob)), p=prob)
            new_loc = within_loc[index]
            new_loc_h3 = utils.cells_to_h3(new_loc,self.resolution)
            new_module = current_module

            usr_each[usr].S_loc[new_loc] += 1

        if keyword == 'cross_return':
            cross_loc = list(usr_each[usr].S_module.keys())
            temp_list = [usr_each[usr].S_module[i] for i in cross_loc]
            prob = np.array(temp_list) / np.sum(temp_list)

            index = np.random.choice(range(len(prob)), p=prob)

            new_loc = cross_loc[index]
            new_loc_h3 = utils.cells_to_h3(new_loc,self.resolution)
            new_module = new_loc

            usr_each[usr].S_module[new_loc] += 1

        usr_each[usr].current_loc = new_loc
        usr_each[usr].current_loc_h3 = new_loc_h3
        usr_each[usr].current_module = new_module
        usr_each[usr].current_t += stay_t
        d_home = usr_each[usr].S_module_d_home[new_module]

        move_r = utils.haversine(current_loc, new_loc)
        move_a = utils.get_bearing(current_loc, new_loc)

        return current_loc_h3, current_loc, new_loc_h3, new_loc, move_r, move_a, stay_t, d_home



    def r_vs_d(self, d_home, inflation_exponent):
        '''
        :param d_home: d_home
        :param inflation_exponent: inflation_exponent
        :return:
        '''
        r = np.power(d_home, inflation_exponent)
        return r


def initialize(user_list, home_list, home_label_list, user_step_list, rgc_exponent):
    usr_Status = {}
    for usr, home, home_label, num_steps in zip(user_list, home_list, home_label_list, user_step_list):
        usr_Status[usr] = UserInfo(usr, home, home_label, num_steps, rgc_exponent)

        usr_Status[usr].S_loc[usr_Status[usr].home] = 1
        usr_Status[usr].S_module[usr_Status[usr].home] = 1
        usr_Status[usr].S_module_loc[usr_Status[usr].home] = [usr_Status[usr].home]
        usr_Status[usr].S_module_d_home[usr_Status[usr].home] = 0.001
        usr_Status[usr].S_module_radius[usr_Status[usr].home] = math.pow(0.001, 0.60)

    return usr_Status

