import sys

import numpy as np

sys.path.append('/Users/luzhong/Documents/pythonCode/20220101Mobility_Project/')
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, explained_variance_score,mean_absolute_percentage_error

from sklearn.metrics import r2_score
from geopy.geocoders import Nominatim
import reverse_geocoder as rg
from scipy.optimize import minimize, rosen, rosen_der
import pickle
####self-defined packages
from input_library import *

import utils
import warnings
warnings.filterwarnings("ignore")

def gravity_model(df_flows,df_home):
    '''
    :param df_flows:
    :param df_home:
    :return: estimated flows between counties
    '''
    df_flows=df_flows[df_flows['month']<=2]
    df_flows=df_flows[df_flows['trips']>1]
    df_flows = df_flows[df_flows['from_label'] !=df_flows['to_label']]

    df_distance=df_flows.groupby(['from_label','to_label'])['distance'].mean().reset_index()

    dis_dict=defaultdict()
    for index,row in df_distance.iterrows():
        if row['from_label'] not in dis_dict.keys():
            dis_dict[row['from_label']]=dict()
        dis_dict[row['from_label']][row['to_label']]=row['distance']

    device_dict=dict(zip(df_home['county'],df_home['devices']))

    df_flows['from_devices']=list(map(lambda x: device_dict[x] if x in device_dict.keys() else 10000, df_flows['from_label']))
    df_flows['to_devices'] = list(map(lambda x: device_dict[x] if x in device_dict.keys() else 10000, df_flows['to_label']))
    df_flows['distance']= list(map(lambda x: dis_dict[x[0]][x[1]], zip(df_flows['from_label'],df_flows['to_label'])))

    print('done with estimation',len(df_flows))
    def linear_fit(x, K,alpha,beta,gamma):
        r_c=np.max(x[:,2])
        r_c=np.mean(x[:,2])
        y=K*np.power(x[:,0],alpha)*np.power(x[:,1],gamma)
        y=y*(np.power(x[:,2],-beta))*np.exp(-x[:,2]/r_c)
        return y

    df_flows_new=[]
    for r in [[0, 100], [100, 200], [200, 500], [500, 1000], [1000, 4000]]:
        df_temp = df_flows[(df_flows['distance'] > r[0]) & (df_flows['distance'] <= r[1])]
        popt, pcov = curve_fit(linear_fit, df_temp[['from_devices','to_devices','distance']].values, df_temp['trips'])
        K,alpha,beta,gamma=popt
        df_temp['flow_syn']=linear_fit(df_temp[['from_devices','to_devices','distance']].values, K,alpha,beta,gamma)
        df_flows_new.append(df_temp)
    df_flows=pd.concat(df_flows_new)
    return df_flows

def extended_radiation_model(df_flows,df_home):
    '''
    :param df_flows:
    :param df_home:
    :return: estimated flows between counties
    '''
    df_flows=df_flows[df_flows['month']==1]
    df_flows=df_flows[df_flows['trips']>1]
    df_flows = df_flows[df_flows['from_label'] !=df_flows['to_label']]


    df_distance=df_flows.groupby(['from_label','to_label'])['distance'].mean().reset_index()

    dis_dict=defaultdict()
    for index,row in df_distance.iterrows():
        if row['from_label'] not in dis_dict.keys():
            dis_dict[row['from_label']]=dict()
        dis_dict[row['from_label']][row['to_label']]=row['distance']

    device_dict=dict(zip(df_home['county'],df_home['devices']))

    df_outflow=df_flows.groupby(['from_label'])['trips'].sum().reset_index()
    outflow_dict = dict(zip(df_outflow['from_label'], df_outflow['trips']))

    df_flows['from_devices']=list(map(lambda x: device_dict[x] if x in device_dict.keys() else 10000, df_flows['from_label']))
    df_flows['to_devices'] = list(map(lambda x: device_dict[x] if x in device_dict.keys() else 10000, df_flows['to_label']))
    df_flows['distance']= list(map(lambda x: dis_dict[x[0]][x[1]], zip(df_flows['from_label'],df_flows['to_label'])))

    df_flows['outflow'] =list(map(lambda x: outflow_dict[x], df_flows['from_label']))
    near_pop=[]
    for i,j in zip(df_flows['from_label'],df_flows['to_label']):
        dis =dis_dict[i][j]
        label_list=[key for key,value in dis_dict[i].items() if value<dis]
        pop_list=[device_dict[k] for k in label_list if k in device_dict.keys()]
        sum_v=np.sum(pop_list)
        near_pop.append(sum_v)

    df_flows['near_pop']=near_pop


    def fun(alpha,df_temp):
        pop_sum=np.unique(list(df_temp['from_label'])+list(df_temp['to_label']))
        pop_sum=df_home[df_home['county'].isin(pop_sum)]['devices'].sum()
        gamma=df_temp['trips'].sum()/pop_sum
        df_temp['flow_syn'] = np.power(df_temp['from_devices'] + df_temp['to_devices'] + df_temp['near_pop'],alpha)
        df_temp['flow_syn']=df_temp['flow_syn'] -np.power(df_temp['from_devices'] +df_temp['near_pop'],alpha)
        df_temp['flow_syn']=df_temp['flow_syn']*(np.power(df_temp['from_devices'],alpha)+1)
        from_sum=df_temp.groupby(['from_label'])['flow_syn'].sum().to_dict()
        #print(df_temp[['flow_syn','from_label']])
        #for index,row in df_temp.iterrows():
            #print(row['flow_syn'],row['from_label'],from_sum[row['from_label']],row['flow_syn']/from_sum[row['from_label']])
        df_temp=df_temp[~df_temp['flow_syn'].isin([np.nan,np.inf,-np.inf])]
        df_temp['flow_syn']=list(map(lambda x:x[0]/(from_sum[x[1]]+10), zip(df_temp['flow_syn'],df_temp['from_label'])))
        df_temp['flow_syn']=df_flows['outflow']*df_temp['flow_syn'] #*gamma
        #####segemented fitting
        return df_temp

    def cal_error(alpha,df_temp):
        df_temp_new=fun(alpha,df_temp)
        error = mean_absolute_percentage_error(df_temp_new['flow_syn'], df_temp_new['trips'])
        return error

    df_list=[]
    for s in [[0,100],[100,200],[200,500],[500,1000],[1000,4000]]: #np.arange(0,40)*100:#
        df_temp = df_flows[(df_flows['distance'] >= s[0]) & (df_flows['distance'] < s[1])]
        res = minimize(cal_error, 0.1, method='Nelder-Mead', args=(df_temp))
        print('distance range:',s,'learned alapha',res.x,res.fun)
        df_temp=fun(res.x[0], df_temp)
        df_list.append(df_temp)

    df_flows=pd.concat(df_list)

    return df_flows

def radiation_model(df_flows,df_home):
    '''
        :param df_flows:
        :param df_home:
        :return: estimated flows between counties
        '''
    df_flows = df_flows[df_flows['month'] == 1]
    df_flows = df_flows[df_flows['trips'] > 1]
    df_flows = df_flows[df_flows['from_label'] != df_flows['to_label']]

    df_distance = df_flows.groupby(['from_label', 'to_label'])['distance'].mean().reset_index()

    dis_dict = defaultdict()
    for index, row in df_distance.iterrows():
        if row['from_label'] not in dis_dict.keys():
            dis_dict[row['from_label']] = dict()
        dis_dict[row['from_label']][row['to_label']] = row['distance']

    device_dict = dict(zip(df_home['county'], df_home['devices']))

    df_outflow = df_flows.groupby(['from_label'])['trips'].sum().reset_index()
    outflow_dict = dict(zip(df_outflow['from_label'], df_outflow['trips']))

    df_flows['from_devices'] = list(
        map(lambda x: device_dict[x] if x in device_dict.keys() else 10000, df_flows['from_label']))
    df_flows['to_devices'] = list(
        map(lambda x: device_dict[x] if x in device_dict.keys() else 10000, df_flows['to_label']))
    df_flows['distance'] = list(map(lambda x: dis_dict[x[0]][x[1]], zip(df_flows['from_label'], df_flows['to_label'])))

    df_flows['outflow'] = list(map(lambda x: outflow_dict[x], df_flows['from_label']))
    near_pop = []
    for i, j in zip(df_flows['from_label'], df_flows['to_label']):
        dis = dis_dict[i][j]
        label_list = [key for key, value in dis_dict[i].items() if value < dis]
        pop_list = [device_dict[k] for k in label_list if k in device_dict.keys()]
        sum_v = np.sum(pop_list)
        near_pop.append(sum_v)

    df_flows['near_pop'] = near_pop
    print(df_flows.head(5))
    df_flows['flow_syn'] = df_flows['outflow'] * df_flows['from_devices'] * df_flows['to_devices']
    df_flows['flow_syn'] = df_flows['flow_syn'] / ((df_flows['from_devices'] + df_flows['near_pop']) * (
                df_flows['from_devices'] + df_flows['to_devices'] + df_flows['near_pop']))

    return df_flows


def visitation_model(df_flows, df_home, df_u):
    '''
    :param df_flows:
    :param df_home:
    :return: estimated flows between counties
    '''
    df_flows = df_flows[df_flows['month'] == 1]
    df_flows = df_flows[df_flows['trips'] >1]

    print('df_flows',len(df_flows))
    df_flows = df_flows[df_flows['from_label'] != df_flows['to_label']]

    df_distance = df_flows.groupby(['from_label', 'to_label'])['distance'].mean().reset_index()

    df_temp=df_flows[df_flows['from_label']!=df_flows['to_label']]
    df_distance_area = df_temp.groupby(['from_label'])['distance'].min().reset_index()
    dis_area_dict = dict(zip(df_distance_area['from_label'],df_distance_area['distance']))
    mean_area=np.mean(list(dis_area_dict.values()))

    print('mean_area',mean_area)

    count=0
    for key,vlue in dis_area_dict.items():
        print(count,key,vlue)
        count+=1
        if count==6:
            break
    dis_dict = defaultdict()

    for index, row in df_distance.iterrows():
        if row['from_label'] not in dis_dict.keys():
            dis_dict[row['from_label']] = dict()
        dis_dict[row['from_label']][row['to_label']] = row['distance']+1



    u_dict=dict(zip(df_u['county'], df_u['u']))
    mean_u=df_u['u'].mean()
    print('mean_u', mean_u)
    df_flows['u'] = list( map(lambda x: u_dict[x] if x in u_dict.keys() else mean_u, df_flows['from_label']))
    df_flows['distance'] = list(map(lambda x: dis_dict[x[0]][x[1]], zip(df_flows['from_label'], df_flows['to_label'])))
    df_flows['area']= list(map(lambda x:dis_area_dict[x]*dis_area_dict[x]*3.1415926 if x in dis_area_dict.keys() else mean_area*3.1415926 ,df_flows['from_label']))
    #df_flows['area'] = list(map(lambda x: dis_area_dict[x] * dis_area_dict[x] * 3.1415926 if x in dis_area_dict.keys() else  3.1415926, df_flows['from_label']))
    df_flows['f'] = list(map(lambda x: np.log((x[0]+1)/x[1]), zip(df_flows['max_f'],df_flows['min_f'])))

    df_flows['flow_syn1']=df_flows['u']*df_flows['area']*df_flows['f'] /(df_flows['distance']*df_flows['distance'])

    df_flows['u'] = list(map(lambda x: u_dict[x] if x in u_dict.keys() else mean_u, df_flows['to_label']))
    df_flows['distance'] = list(map(lambda x: dis_dict[x[0]][x[1]], zip(df_flows['from_label'], df_flows['to_label'])))
    df_flows['area'] = list(map(lambda x:dis_area_dict[x]*dis_area_dict[x] if x in dis_area_dict.keys() else mean_area*3.1415926 , df_flows['from_label']))
    #df_flows['area'] = list(map(lambda x: dis_area_dict[x] * dis_area_dict[x] * 3.1415926 if x in dis_area_dict.keys() else 3.1415926, df_flows['from_label']))

    df_flows['f'] = list(map(lambda x: np.log((x[0]+1)/x[1]), zip(df_flows['max_f'],df_flows['min_f'])))

    df_flows['flow_syn2'] = df_flows['u'] * df_flows['area'] * df_flows['f'] / (df_flows['distance'] * df_flows['distance'])

    df_flows['flow_syn']=df_flows['flow_syn1'] #df_flows['flow_syn2']+

    def linear_fit(x, a):
        return a * x

    print(df_flows['flow_syn'].max())
    popt, pcov = curve_fit(linear_fit, df_flows['flow_syn'], df_flows['trips'])
    print('learned para',popt[0])
    df_flows['flow_syn'] = df_flows['flow_syn'] #* popt[0]+
    return df_flows



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
    oringal_count=len(df_home)
    df_home = df_home[~df_home['home_lon'].isna()]
    true_count = len(df_home)

    df_home=city_state_country(df_home, 'home_lat', 'home_lon')
    df_home=df_home[df_home['country']=='US']
    df_home_devices=df_home.groupby(['county','state','country'])['id'].count().reset_index()
    df_home_devices['devices']=df_home_devices['id']*oringal_count/true_count
    df_home_devices=df_home_devices[['county','state','country','devices']]

    return df_home_devices

def accuracy_with_dis(df_compare):
    '''
    try with different accury metric
    :param df_compare:E
    :return: accuracy of estimation in increasing spatial scale
    '''
    k=1/4
    dis_list=np.arange(1,400)*10
    #dis_list = np.arange(1, 10) * 10
    dis_from=[i-i*k for i in dis_list]
    dis_to = [i +10+i * k for i in dis_list]
    output_mat=[]
    for from_v, dis_v, to_v in zip(dis_from,dis_list,dis_to):
        df_temp=df_compare[(df_compare['distance']>from_v)&(df_compare['distance']<=to_v)]
        error=mean_absolute_percentage_error(df_temp['flow_syn'],df_temp['trips'])
        ssi=np.mean([2*min(i,j)/(i+j) for i,j in zip(df_temp['flow_syn'],df_temp['trips'])])
        r = explained_variance_score(df_temp['flow_syn'], df_temp['trips'])
        output_mat.append([dis_v,error, r ,ssi])

    df_error = pd.DataFrame(output_mat, columns=['distance', 'MSE','Variance','SSI'])

    return  df_error

def input_data(case):
    df_flows = pd.read_csv('/Volumes/SeagateDrive/Mobility_Project/stoppoints_analyze_results/county_flowsum.csv')
    df_home = pd.read_csv('/Volumes/SeagateDrive/Mobility_Project/stoppoints_analyze_results/all_usr_home_sum.csv')
    df_home_devices = each_loc_devices(df_home)
    df_home_devices.to_csv('/Volumes/SeagateDrive/Mobility_Project/stoppoints_analyze_results/home_devices.csv')
    if case=='Gravity model':
        path_result = '/Users/lucinezhong/Documents/LuZHONGResearch/20210328Scale_Mobility/Results_flow/Gravity_model/'
        return df_flows,df_home_devices,path_result


    if case=='visitation':
        print(case)
        typex='resolution'
        typex ='county'
        df_home = pd.read_csv('/Volumes/SeagateDrive/Mobility_Project/stoppoints_analyze_results/all_usr_home_sum.csv')
        df_home_devices = each_loc_devices(df_home)
        print('home_device_done')

        df_flows = pd.read_csv('/Volumes/SeagateDrive/Mobility_Project/stoppoints_analyze_results/'+typex+'_flowsum.csv')
        print('flows done')
        if typex=='resolution':
            import h3
            boston_lat = 42.361145
            boston_lon = -71.057083
            origin = h3.geo_to_h3(lat=boston_lat, lng=boston_lon, resolution=7)
            all_index = h3.k_ring(origin, 100)
            df_flows=df_flows[(df_flows['from_label'].isin(all_index))&(df_flows['to_label'].isin(all_index))]
            print('selected flows',len(df_flows))

        path_result = '/Users/lucinezhong/Documents/LuZHONGResearch/20210328Scale_Mobility/Results_flow/'
        df_frequency=pd.read_csv('/Volumes/SeagateDrive/Mobility_Project/stoppoints_analyze_results/'+typex+'max_min_f.csv')
        df_frequency['from_label']=df_frequency['home_label']
        df_frequency['to_label'] = df_frequency['label']
        df_flows=df_flows.merge(df_frequency,how='left',on=['from_label','to_label'])
        print(np.max(df_flows['max_f']))
        df_flows['max_f']=list(map(lambda x: 1 if x==None or np.isnan(x)==True else x, df_flows['max_f']))
        df_flows['min_f'] =list(map(lambda x: 1 if x==None or np.isnan(x)==True else x, df_flows['min_f']))

        print(np.max(df_flows['max_f']))
        print(df_flows.head(5))

        df_u=pd.read_csv('/Users/lucinezhong/Documents/LuZHONGResearch/20210328Scale_Mobility/Results_visitation/learned_'+typex+'_u.csv')

        return df_flows, df_home_devices, path_result,df_u

def flow_prob_distance():
    path_result = '/Volumes/SeagateDrive/Mobility_Project/stoppoints_analyze_results/'
    df_flows = pd.read_csv('/Volumes/SeagateDrive/Mobility_Project/stoppoints_analyze_results/county_flowsum.csv')

    df_flows = df_flows[df_flows['month'] <= 2]
    df_flows = df_flows[df_flows['trips'] > 1]

    neighbour_sum = df_flows.groupby(['from_label'])['trips'].sum().reset_index()
    neighbour_sum_dict = dict(zip(neighbour_sum['from_label'], neighbour_sum['trips']))

    df_temp=df_flows[df_flows['from_label']==df_flows['to_label']]
    self_sum = df_temp.groupby(['from_label'])['trips'].sum().reset_index()
    self_sum_dict = dict(zip(self_sum['from_label'], self_sum['trips']))

    mean_self=[self_sum_dict[key]/value for key,value in neighbour_sum_dict.items() if self_sum_dict[key]/value<0.9]
    print('ave',np.mean(mean_self),np.median(mean_self))
    mean_self=np.mean(mean_self)

    loc_loc_prob = defaultdict()
    loc_loc_distance= defaultdict()
    for from_v in pd.unique(df_flows['from_label']):
        df_temp = df_flows[df_flows['from_label'] == from_v]
        if from_v not in list(df_temp['to_label']):
            dict_temp = dict(zip(df_temp['to_label'], (1-mean_self)* df_temp['trips'] / neighbour_sum_dict[from_v]))
            loc_loc_prob[from_v] = dict_temp
            loc_loc_prob[from_v][from_v] = mean_self
        else:
            dict_temp = dict(zip(df_temp['to_label'], df_temp['trips'] / neighbour_sum_dict[from_v]))
            loc_loc_prob[from_v]=dict_temp
        loc_loc_distance[from_v] = dict(zip(df_temp['to_label'], df_temp['distance']))

    with open(path_result+'loc_loc_prob.pickle', 'wb') as handle:
        pickle.dump(loc_loc_prob, handle, protocol=pickle.HIGHEST_PROTOCOL)

    df_county_geo=pd.read_csv('/Volumes/SeagateDrive/Mobility_Project/stoppoints_analyze_results/county_demographic.csv')
    df_county_geo['Latitude'] =df_county_geo['Latitude'].astype(float)
    df_county_geo['Longitude'] = df_county_geo['Longitude'].astype(float)
    pos_list=list(zip(df_county_geo['Latitude'],df_county_geo['Longitude']))
    county_pos=dict(zip(df_county_geo['County_name'],pos_list))
    pos_county = dict(zip(pos_list,df_county_geo['County_name']))

    with open(path_result+'county_pos.pickle', 'wb') as handle:
        pickle.dump(county_pos, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path_result+'pos_county.pickle', 'wb') as handle:
        pickle.dump(pos_county, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for key1, value1 in county_pos.items():
        if key1 not in loc_loc_distance.keys():
            loc_loc_distance[key1]=dict()
        for key2, value2 in county_pos.items():
            loc_loc_distance[key1][key2] = utils.haversine(value1, value2)

    with open(path_result+'loc_loc_distance.pickle', 'wb') as handle:
        pickle.dump(loc_loc_distance, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    case='flow_prob&distance'
    #case='Gravity model'
    #case='Radiation model'
    #case = 'Extended radiation model'
    #case='visitation'
    ####to check the accuray results, seee model_accuracy.py
    if case=='flow_prob&distance':
        flow_prob_distance()

    if case=='Gravity model':
        df_flows, df_home_devices, path_result = input_data(case)
        df_compare=gravity_model(df_flows,df_home_devices)
        df_compare.to_csv(path_result + case + 'flow.csv')

    if case=='Radiation model':
        df_flows, df_home_devices, path_result = input_data(case)
        df_compare = radiation_model(df_flows, df_home_devices)
        df_compare.to_csv(path_result + case + 'flow.csv')

    if case == 'Extended radiation model':
        df_flows, df_home_devices, path_result = input_data(case)
        df_compare = extended_radiation_model(df_flows, df_home_devices)
        df_compare.to_csv(path_result + case + 'flow.csv')


    if case=='visitation':
        df_flows, df_home_devices, path_result, df_u= input_data(case)
        df_compare = visitation_model(df_flows, df_home_devices, df_u)
        df_compare.to_csv(path_result + case + 'flow.csv')




