import sys
sys.path.append(
    '/Users/lucinezhong/Documents/LuZHONGResearch/20210328Scale_Mobility/202407network_project/network_code/network_mobility_python_code/')

import h3
import math
from collections import defaultdict
import numpy as np
import pandas as pd

import utils
import warnings
warnings.filterwarnings("ignore")


resolution=10

class UserInfo:
    def __init__(self, id_num, home_lat_lon,hm_label,step_num):
        self.id = id_num
        self.home =(round(home_lat_lon[0],3),round(home_lat_lon[1],3))
        self.home_label =cells_to_h3(home_lat_lon)

        self.current_loc=self.home
        self.current_loc_label= self.home_label

        self.new_loc = -1
        self.new_loc_county=-1

        self.current_t = 0
        self.step_num = step_num
        self.step_time= step_num * 60
        ####saved location and visit frequency

        self.S_loc = dict()  ######visited locations and their frequency
        self.S_label=dict()


class region_grid:
    def __init__(self):
        self.loc_label=defaultdict()


def GR_powerlaws(para,min_v,max_v,num):
    '''
        :param alpha: exponent of power law
        :papra num: number of random values
        return random values
        '''
    x0=min_v
    x1=max_v
    y = np.random.uniform(0, 1, num)
    x = np.power((math.pow(x1, para + 1) - math.pow(x0, para + 1)) * y + math.pow(x0, para + 1), 1 / (para + 1))
    return x

def GR_random( min_v,max_v,num):
    x = np.random.uniform(min_v, max_v, num)
    return x

class EPR_model():
    def __init__(self, user_list, rho,gamma, beta_r, beta_t):
        self.user_list = user_list
        self.verbose = False

        self.beta_r = beta_r
        self.beta_t = beta_t

        self.rho=rho
        self.gamma=gamma

    def simulation(self,usr_Status,region_Status):
        """
        :param alpha: exponent for stay duration
        :param beta: exponent for distance
        :param rho: explore para
        :param gamma: exponent for visited locations
        :param home_lat: initial lat
        :param home_lon: initial lon
        :param timestep: simualiton steps
        :param str_individual: individual
        :return: individual trajectory dataframe
        """
        output_mat = []
        print('total_user',len(self.user_list))
        for cnt, usr in enumerate(self.user_list):
            print('cnt,usr',cnt,usr,usr_Status[usr].step_num)
            for step in range(0, usr_Status[usr].step_num):
                #if usr_Status[usr].current_t <= self.time_step:
                #####come to the time when the user need to make a decision
                temp = np.random.rand()
                P_new = self.rho * math.pow(len(usr_Status[usr].S_loc), self.gamma)
                ###Explore
                if temp < P_new:
                    keyword = 'explore'
                    current_label,current_loc,new_label,new_loc,move_r,move_a,stay_t,d_home= self.Explore_location(usr,usr_Status,region_Status)
                   # print(usr,step,current_label,new_label)
                else:
                    keyword = 'return'
                    current_label,current_loc,new_label,new_loc,move_r,move_a,stay_t,d_home= self.Return_location(usr,usr_Status, region_Status)

                output_mat.append([usr, usr_Status[usr].home_label, step, keyword, current_label, new_label, current_loc[0], current_loc[1], new_loc[0], new_loc[1], move_r, move_a, stay_t, usr_Status[usr].current_t, d_home])
            #####keeping exploring at neighbours##############
        output_mat = np.array(output_mat, dtype=object)
        columns = ['id', 'home_label', 'step', 'keyword', 'from_label', 'to_label', 'from_lat', 'from_lon',
                   'to_lat', 'to_lon', 'travel_d(km)', 'travel_angle', 'stay_t(h)', 'start', 'd_home']

        df = pd.DataFrame(data=output_mat, columns=columns)
        return df

    def Explore_location(self,usr,usr_each,region_Status):
        '''
        :param lat,lon: current locations' lat and lon
        :papra d: move distance
        :para brng: move angle
        return locations' lat,lon
        '''

        current_loc = usr_each[usr].current_loc
        current_label = usr_each[usr].current_loc_label
        move_r = GR_powerlaws(self.beta_r,0.03, 4000,1)[0]
        move_a = GR_random(-180, 180, 1)[0]
        stay_t = GR_powerlaws(self.beta_t,10, 24 * 60,1)[0]

        ####begin find new location

        new_loc = self.find_move_loc(current_loc, move_r, move_a)
        new_label = self.save_loc_h3(new_loc, region_Status)

        if new_loc not in usr_each[usr].S_loc.keys():
            usr_each[usr].S_loc[new_loc] = 1
        else:
            usr_each[usr].S_loc[new_loc] = +1

        if new_label not in usr_each[usr].S_label.keys():
            usr_each[usr].S_label[new_label] = 1
        else:
            usr_each[usr].S_label[new_label] += 1

        region_Status.loc_label[new_loc]=new_label

        usr_each[usr].current_loc = new_loc
        usr_each[usr].current_loc_label = new_label
        usr_each[usr].current_t += stay_t*2

        move_r =utils.haversine(current_loc, new_loc)
        move_a = utils.get_bearing(current_loc, new_loc)
        d_home = utils.haversine(usr_each[usr].home, new_loc)
        return current_label,current_loc,new_label,new_loc,move_r,move_a,stay_t,d_home

    def Return_location(self,usr,usr_each,region_Status):
        '''
        :param S: visited locations
        :papra prob: locations' probablity
        return locations' lat,lon,index
        '''
        current_label = usr_each[usr].current_loc_label
        current_loc = usr_each[usr].current_loc

        temp_list = list(usr_each[usr].S_loc.values())
        prob = np.array(temp_list) / np.sum(temp_list)
        ####return home or not
        index = np.random.choice(range(len(prob)), p=prob)
        # print('temp_list',temp_list)
        new_loc = list(usr_each[usr].S_loc.keys())[index]
        new_label = cells_to_h3(new_loc)
        stay_t = GR_powerlaws(self.beta_t,10, 24 * 60,1)[0]

        usr_each[usr].S_loc[new_loc] += 1
        usr_each[usr].S_label[new_label] += 1
        usr_each[usr].current_loc = new_loc
        usr_each[usr].current_loc_label = new_label
        usr_each[usr].current_t += stay_t*2

        move_r = utils.haversine(current_loc, new_loc)
        move_a = utils.get_bearing(current_loc, new_loc)
        d_home = utils.haversine(usr_each[usr].home, new_loc)

        #print('return', current_label, new_label)
        return current_label,current_loc,new_label,new_loc,move_r,move_a,stay_t,d_home

    def find_move_loc(self,pos,move_r,move_a):
        R = 6378.1  # Radius of the Earth
        (lat,lon)=pos
        lat1 = math.radians(lat)  # Current lat point converted to radians
        lon1 = math.radians(lon)  # Current long point converted to radians

        lat2 = math.asin(math.sin(lat1) * math.cos(move_r / R) +
                         math.cos(lat1) * math.sin(move_r / R) * math.cos(move_a))
        lon2 = lon1 + math.atan2(math.sin(move_a) * math.sin(move_r / R) * math.cos(lat1),
                                 math.cos(move_r / R) - math.sin(lat1) * math.sin(lat2))

        lat2 =math.degrees(lat2)
        lon2 = math.degrees(lon2)
        return (round(lat2,3),round(lon2,3))

    def save_loc_h3(self, new_loc, region_Status):
        if new_loc not in region_Status.loc_label.keys():
            new_label = cells_to_h3(new_loc)
            region_Status.loc_label[new_loc] = new_label
        else:
            new_label = region_Status.loc_label[new_loc]

        return new_label

def initialize(user_list, home_list, home_label_list, user_step_list):
    region_Status = region_grid()

    usr_Status = {}
    for usr, home, hm_label,num_steps in zip(user_list, home_list, home_label_list,user_step_list):
        usr_Status[usr] = UserInfo(usr, home, hm_label, num_steps)
        usr_Status[usr].S_loc[home] = 1
        usr_Status[usr].S_label[cells_to_h3(home)] = 1

        region_Status.loc_label[home] = cells_to_h3(home)

    return usr_Status, region_Status


def cells_to_h3(pos):
    new_label = h3.geo_to_h3(pos[0], pos[1], resolution=resolution)
    return new_label

def cells_to_county(pos):
    coordinates = pos
    location = rg.search(coordinates)
    new_label = location[0]['admin1'] + "_" + location[0]['admin2']
    return new_label
