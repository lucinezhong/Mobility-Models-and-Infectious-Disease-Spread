import math
import sys
sys.path.append(
    '/Users/lucinezhong/Documents/LuZHONGResearch/20210328Scale_Mobility/202407network_project/network_code/network_mobility_python_code/')

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, explained_variance_score,mean_absolute_percentage_error,r2_score
from collections import defaultdict
import h3

####self-defined packages
import utils

import warnings
warnings.filterwarnings("ignore")

resolution=10

check_inflation=True
check_module_return_home=True

class UserInfo:
    def __init__(self, id_num, home_lat_lon,hm_label,step_num,rgc_slope):
        self.id = id_num
        self.home =home_lat_lon
        self.home_label =hm_label

        self.current_loc=self.home
        self.current_loc_h3=cells_to_h3(self.home)
        self.current_module=self.current_loc

        self.new_loc=-1
        self.new_loc_h3=-1
        self.new_module=-1

        self.current_t = 0
        self.step_num = step_num
        self.step_time = step_num * 60

        self.S_loc = dict()  ######visited locations and their frequency

        #####save_explore_cluster_frequncy
        self.S_module_d_home = dict()
        self.S_module_loc=dict()
        self.S_module=dict() ######visited modules and their frequency
        self.S_module_radius=dict()

        self.rgc=GR_powerlaws(rgc_slope,1,4000,1)[0]


class region_grid:
    def __init__(self):
        self.loc_county= defaultdict()


class Switch_model():
    def __init__(self,user_list,prob_switch,gamma_w,gamma_c_slope,rho_w,rho_c,beta_r,beta_t):
        self.user_list=user_list
        self.verbose=False

        self.beta_r=beta_r
        self.beta_t=beta_t

        self.prob_switch = prob_switch
        self.rho_w=rho_w
        self.gamma_w =gamma_w
        self.rho_c =rho_c
        self.gamma_c_slope=gamma_c_slope
        self.slope_r_d=0.60


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

        #####parameters setting
        output_mat=[]
        print(len(self.user_list))
        for cnt, usr in enumerate(self.user_list):
            print('cnt,usr',cnt,usr,usr_Status[usr].step_num)
            for step in range(0, usr_Status[usr].step_num):
                #####come to the time when the user need to make a decision

                current_module=usr_Status[usr].current_module
                current_d_home = usr_Status[usr].S_module_d_home[current_module]

                S_w_loc=len(np.unique(usr_Status[usr].S_module_loc[current_module]))
                P_w = self.rho_w* math.pow(S_w_loc,self.gamma_w) ##0.47
                #print(current_d_home,'old_P_w',P_w)

                S_c_loc=len(list(usr_Status[usr].S_module.keys()))
                new_gamma_c=-0.6+math.log10(usr_Status[usr].rgc+1)*self.gamma_c_slope
                if new_gamma_c>0:
                    new_gamma_c=0

                P_c= self.rho_c*math.pow(S_c_loc, new_gamma_c)

                P_switch=self.prob_switch

                #P_switch=self.prob_switch-np.log10(usr_Status[usr].rgc + 1)*0.025
                #####wihtin exploration scope

                temp = np.random.rand()
                if temp<1-P_switch and S_w_loc<=10:
                    temp = np.random.rand()
                    if temp <= P_w:
                        keyword = 'within_explore'
                        current_h3, current_loc, new_h3, new_loc, move_r, move_a, stay_t, d_home = self.Explore(usr,
                                                                                                                usr_Status,
                                                                                                                region_Status,
                                                                                                                keyword)

                    else:
                        keyword = 'within_return'
                        current_h3, current_loc, new_h3, new_loc, move_r, move_a, stay_t, d_home = self.Return(usr,
                                                                                                               usr_Status,
                                                                                                               region_Status,
                                                                                                               keyword)
                else:
                    #####cross exploration scope
                    temp = np.random.rand()
                    if temp < P_c:
                        keyword = 'cross_explore'
                        current_h3, current_loc, new_h3, new_loc, move_r, move_a, stay_t, d_home = self.Explore(usr,
                                                                                                                usr_Status,
                                                                                                                region_Status,
                                                                                                                keyword)

                    else:
                        keyword = 'cross_return'
                        current_h3, current_loc, new_h3, new_loc, move_r, move_a, stay_t, d_home = self.Return(
                            usr, usr_Status, region_Status, keyword)

                #print(usr,keyword,move_r,step)
                output_mat.append([usr, usr_Status[usr].home_label,step, keyword, current_h3,new_h3,current_loc[0],current_loc[1],new_loc[0],new_loc[1],move_r,move_a,stay_t,usr_Status[usr].current_t,d_home])
                #####keeping exploring at neighbours##############
        output_mat = np.array(output_mat, dtype=object)
        columns = ['id', 'home_label','step','keyword',  'from_label', 'to_label', 'from_lat','from_lon','to_lat','to_lon','travel_d(km)','travel_angle','stay_t(h)','start','d_home']

        df = pd.DataFrame(data=output_mat, columns=columns)
        return df

    def Explore(self,usr,usr_each,region_Status,keyword):
        '''
        :param lat,lon: current locations' lat and lon
        :papra d: move distance
        :para brng: move angle
        return locations' lat,lon
        '''

        current_loc = usr_each[usr].current_loc
        current_loc_h3 = usr_each[usr].current_loc_h3
        current_module = usr_each[usr].current_module
        current_d_home=usr_each[usr].S_module_d_home[current_module]
        current_module_r=usr_each[usr].S_module_radius[current_module]

        move_a = GR_random(-180,180,1)[0]
        stay_t = GR_powerlaws(self.beta_t,10, 24 * 60,1)[0]

        ####begin find new location
        if keyword=='within_explore':
            #move_r = GR_powerlaws(self.beta_r, 0.03, current_module_r, 1)[0]
            move_r = GR_powerlaws(self.beta_r, 0.03, 4000, 1)[0]
            new_loc = self.find_move_loc(current_loc, move_r, move_a) ####
            new_loc_h3 = cells_to_h3(new_loc)
            new_module = current_module
            usr_each[usr].S_module_loc[new_module].append(new_loc)

            if self.verbose==True:
                new_label = self.save_loc_county(new_loc, region_Status)
                print('within_explore',new_loc,new_label)

        if keyword=='cross_explore':
            #move_r =GR_powerlaws(self.beta_r,current_module_r,4000,1)[0]
            #move_r = GR_powerlaws(-1.2, current_module_r, 4000, 1)[0]
            move_r = GR_powerlaws(-1.2, 0.03, 4000, 1)[0]
            new_loc = self.find_move_loc(current_loc, move_r, move_a)
            new_loc_h3 = cells_to_h3(new_loc)
            new_module = new_loc
            if self.verbose == True:
                new_label = self.save_loc_county(new_loc, region_Status)
                print('cross_explore',new_loc,new_label)

            ###explore a new location
            usr_each[usr].S_module[new_loc] = 1  #####frequency
            usr_each[usr].S_module_loc[new_loc]=[new_loc] #####loc squence
            usr_each[usr].S_module_d_home[new_loc] = utils.haversine(usr_each[usr].home, new_loc)

            if check_inflation==True:
                r_cluster = self.r_vs_d(usr_each[usr].S_module_d_home[new_loc]  + 1, self.slope_r_d)
            else:
                r_cluster=10
            usr_each[usr].S_module_radius[new_loc]=r_cluster


        usr_each[usr].current_loc = new_loc
        usr_each[usr].current_loc_h3=new_loc_h3
        usr_each[usr].current_module=new_module
        usr_each[usr].S_loc[new_loc] = 1
        usr_each[usr].current_t +=stay_t*2
        d_home=usr_each[usr].S_module_d_home[new_module]

        return current_loc_h3,current_loc,new_loc,new_loc,move_r,move_a,stay_t,d_home

    def Return(self,usr,usr_each,region_Status,keyword):
        '''
        :param S: visited locations
        :papra prob: locations' probablity
        return locations' lat,lon,index
        '''

        current_loc = usr_each[usr].current_loc
        current_loc_h3 = usr_each[usr].current_loc_h3
        current_module = usr_each[usr].current_module
        stay_t = GR_powerlaws(self.beta_t, 10, 24 * 60, 1)[0]


        if keyword == 'within_return':
            within_loc=usr_each[usr].S_module_loc[current_module]

            if len(within_loc)>1:
                temp_list = [usr_each[usr].S_loc[i] for i in within_loc]
                prob = np.array(temp_list) / np.sum(temp_list)
                index = np.random.choice(range(len(prob)), p=prob)
                new_loc = within_loc[index]
                new_loc_h3 = cells_to_h3(new_loc)
                new_module=current_module
            else:
                new_loc = current_loc
                new_loc_h3 = current_loc_h3
                new_module = current_module

            usr_each[usr].S_loc[new_loc] += 1
            if self.verbose == True:
                new_label = self.save_loc_county(new_loc_h3, region_Status)
                print('within_return', new_loc, new_label)


        if keyword=='cross_return':
            cross_loc = list(usr_each[usr].S_module.keys())

            if check_module_return_home==True:
                index=0   ####return home
            else:
                temp_list = [usr_each[usr].S_module[i] for i in cross_loc]
                prob = np.array(temp_list) / np.sum(temp_list)

                index = np.random.choice(range(len(prob)), p=prob)

            new_loc = cross_loc[index]
            new_loc_h3 = cells_to_h3(new_loc)
            new_module = new_loc

            usr_each[usr].S_module[new_loc]+=1

            if self.verbose == True:
                new_label = self.save_loc_county(new_loc_h3, region_Status)
                print('cross_return', new_loc, new_label)

        usr_each[usr].current_loc = new_loc
        usr_each[usr].current_loc_h3 = new_loc_h3
        usr_each[usr].current_module = new_module
        usr_each[usr].current_t += stay_t*2
        d_home = usr_each[usr].S_module_d_home[new_module]

        move_r = utils.haversine(current_loc, new_loc)
        move_a = utils.get_bearing(current_loc, new_loc)

        return current_loc_h3,current_loc,new_loc_h3,new_loc,move_r,move_a,stay_t,d_home


    def save_loc_county(self, current_label, region_Status):
        if current_label not in region_Status.loc_county.keys():
            new_pos = h3.h3_to_geo(current_label)
            new_label=cells_to_county(new_pos)
            region_Status.loc_county[current_label] = new_label
        else:
            new_label = region_Status.loc_county[current_label]

        return new_label

    

    def find_move_loc(self,pos,move_r,move_a):
        R = 6378.1  # Radius of the Earth
        (lat,lon)=pos
        lat1 = math.radians(lat)  # Current lat point converted to radians
        lon1 = math.radians(lon)  # Current long point converted to radians

        lat2 = math.asin(math.sin(lat1) * math.cos(move_r / R) +
                         math.cos(lat1) * math.sin(move_r / R) * math.cos(move_a))
        lon2 = lon1 + math.atan2(math.sin(move_a) * math.sin(move_r / R) * math.cos(lat1),
                                 math.cos(move_r / R) - math.sin(lat1) * math.sin(lat2))

        lat2 = round(math.degrees(lat2),3)
        lon2 = round(math.degrees(lon2),3)
        return (lat2,lon2)

    def r_vs_d(self,d_home,slope_r_d):
        ###input of inflation pattern
        r=np.power(d_home,slope_r_d)
        return r

def initialize(user_list, home_list,home_label_list,user_step_list,rgc_slope):
    usr_Status = {}
    for usr,home,hm_label,num_steps in zip(user_list,home_list,home_label_list,user_step_list):
        usr_Status[usr] = UserInfo(usr, home,hm_label,num_steps,rgc_slope)
        usr_Status[usr].S_loc[usr_Status[usr].home]=1
        usr_Status[usr].S_module[usr_Status[usr].home] = 1
        usr_Status[usr].S_module_loc[usr_Status[usr].home]=[usr_Status[usr].home]
        usr_Status[usr].S_module_d_home[usr_Status[usr].home] = 0.001
        usr_Status[usr].S_module_radius[usr_Status[usr].home]=math.pow(0.001,0.60)

    region_Status=region_grid()
    return usr_Status,region_Status



def cells_to_h3(pos):
    new_label = h3.geo_to_h3(pos[0], pos[1], resolution=resolution)
    return new_label

def cells_to_county(pos):
    coordinates = pos
    location = rg.search(coordinates)
    new_label = location[0]['admin1'] + "_" + location[0]['admin2']
    return new_label

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


def city_state_country(df, lat_str, lon_str):
    coordinates = list(zip(df[lat_str], df[lon_str]))
    location = rg.search(coordinates)
    df['county'] = list(map(lambda x: x['admin1'] + "_" + x['admin2'], location))
    df['state'] = list(map(lambda x: x['admin1'], location))
    df['country'] = list(map(lambda x: x['cc'], location))
    return df

def each_loc_devices(df_home):
    '''
    :param df_home:
    :return: devices in different home location
    '''
    df_home['home_lat'] = df_home['home_lat'].astype(float)
    df_home['home_lon'] = df_home['home_lon'].astype(float)
    df_home['home_lat']=df_home['home_lat'].round(3)
    df_home['home_lon'] = df_home['home_lon'].round(3)
    oringal_count=len(df_home)
    df_home = df_home[~df_home['home_lon'].isna()]
    true_count = len(df_home)

    df_home=city_state_country(df_home, 'home_lat', 'home_lon')
    df_home=df_home[df_home['country']=='US']
    df_home_devices=df_home.groupby(['county','state','country'])['id'].count().reset_index()
    df_home_devices['devices']=df_home_devices['id']*oringal_count/true_count
    df_home_devices=df_home_devices[['county','state','country','devices']]

    return df_home,df_home_devices


def city_state_country(df, lat_str, lon_str,name_str):
    coordinates = list(zip(df[lat_str], df[lon_str]))
    location = rg.search(coordinates)
    df[name_str] = list(map(lambda x: x['admin1'] + "_" + x['admin2'], location))
    return df


def groupby_flow_county(df_raw):

    df_flows = pd.read_csv('/Volumes/SeagateDrive/Mobility_Project/stoppoints_analyze_results/county_flowsum(groups)_processed.csv')
    df_flows['E[d]'] = df_flows['E[d]'].replace(10000, 1000)
    #print(df_flows.columns)
    df = copy.deepcopy(df_raw)
    df['E[d]'] = df['E[d]'].replace(10000, 1000)
    df=df[(df['from_lat']!=df['to_lat'])&(df['from_lon']!=df['to_lon'])]
    df['flow_syn'] = [1] * len(df)
    df['distance']=df['travel_d(km)']

    df=city_state_country(df, 'from_lat', 'from_lon','from_label')
    df = city_state_country(df, 'to_lat', 'to_lon', 'to_label')


    df_group=df.groupby(['from_label','to_label','iter','E[d]']).agg({'flow_syn':['count'],'id':['nunique']}).reset_index()
    df_group.columns = df_group.columns.droplevel(1)
    df_group.columns=['from_label','to_label','iter','E[d]','flow_syn','unique_usrs']
    df_group=df_group.groupby(['from_label','to_label','E[d]']).agg({'flow_syn':['mean'],'unique_usrs':['mean']}).reset_index()
    df_group.columns = df_group.columns.droplevel(1)
    df_group.columns = ['from_label', 'to_label','E[d]', 'flow_syn', 'unique_usrs']

    df_flows = df_flows[df_flows['month'] <=2]
    #print(df_flows.columns)
    df_flows = df_flows.groupby(['from_label','to_label','E[d]']).agg({'trips': ['sum'], 'unique_usrs': ['sum'],'distance':['mean']}).reset_index()
    df_flows.columns=['from_label','to_label','E[d]','trips','unique_usrs','distance']

    df_group=df_group.merge(df_flows,on=['from_label','to_label','E[d]'],how='right')

    #df_group['distance']=df_group['distance_y']
    df_group = df_group[df_group['from_label'].str.contains('New York_Albany County')]

    return df_group


def groupby_flow_home_destination(df_raw):
    df_flows = pd.read_csv('/Volumes/SeagateDrive/Mobility_Project/stoppoints_analyze_results/county_flowsum_home_destination(groups)_processed.csv')
    #print(df_flows.columns)
    df_flows['E[d]'] = df_flows['E[d]'].replace(10000, 1000)
    df = copy.deepcopy(df_raw)
    df['E[d]'] = df['E[d]'].replace(10000, 1000)
    df = df[(df['from_lat'] != df['to_lat']) & (df['from_lon'] != df['to_lon'])]
    df['flow_syn'] = [1] * len(df)
    df['distance'] = df['travel_d(km)']

    df = city_state_country(df, 'to_lat', 'to_lon', 'to_label')

    df_group = df.groupby(['home_label', 'to_label', 'iter', 'E[d]']).agg({'flow_syn': ['count'], 'id': ['nunique']}).reset_index()
    df_group.columns = df_group.columns.droplevel(1)
    df_group.columns = ['home_label', 'to_label', 'iter', 'E[d]', 'flow_syn', 'unique_usrs']
    df_group = df_group.groupby(['home_label', 'to_label', 'E[d]']).agg( {'flow_syn': ['mean'], 'unique_usrs': ['mean']}).reset_index()
    df_group.columns = df_group.columns.droplevel(1)
    df_group.columns = ['from_label', 'to_label', 'E[d]', 'flow_syn', 'unique_usrs']

    df_flows = df_flows[df_flows['month'] <= 2]
    df_flows['distance']=df_flows['distance'].astype(float)
    df_flows = df_flows.groupby(['from_label', 'to_label','E[d]']).agg(
        {'trips': ['sum'], 'unique_usrs': ['sum'], 'distance': ['mean']}).reset_index()
    df_flows.columns = ['from_label', 'to_label','E[d]','trips', 'unique_usrs', 'distance']

    df_group = df_group.merge(df_flows, on=['from_label', 'to_label','E[d]'], how='right')

    df_group = df_group[df_group['from_label'].str.contains('New York_Albany County')]  #
    return df_group
