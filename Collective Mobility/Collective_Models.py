
import numpy as np
import pandas as pd
import utils
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import curve_fit

class Collective_Model:
    def __init__(self, df_flow):
        self.df_flow=df_flow

    def Gravity_Model(self,X,K, alpha, gamma, beta):
        '''
        :param X: X=[m_i,m_j,r_ij]; including local population of mi, mj, and their distance r_ij
        :param K: constant
        :return: estimated flows
        '''

        T_ij = K * np.power(X[:, 0], alpha) * np.power(X[:, 1], gamma)
        f_rij = np.power(X[:, 2], beta)
        T_ij = T_ij /f_rij
        return T_ij

    def Radiation_Model(self,X):
        '''
        :param X: X=[m_i,m_j,T_i, S_ij]; including local population of mi, mj, total outflow T_i and S_ij, the population around r_ij of location i
        :return: estimated flows
        '''

        T_ij=X[:, 0]*X[:, 1]*X[:, 2]
        T_ij=T_ij/((X[:, 0]+X[:, 3])*(X[:, 0]+X[:, 1]+X[:, 3]))
        return T_ij

    def Visitation_Model(self,X):
        '''
        :param X: X=[A_i,u_j, r_ij]; including attractiveness of location j; area of location i, and distance r_ij
        :return: estimated flows
        '''
        T_ij=X[:,0]*X[:,1]
        T_ij=T_ij/(X[:,2]*X[:,2]*np.log(X[:,3]))
        return T_ij

    def compute_near_pop(self, from_label, dis):
        '''
        :param from_label: source regions of flow
        :param dis: distance between source region and destination region
        :return:
        '''
        df_temp = self.df_flow[(self.df_flow['from_label'] == from_label)&(self.df_flow['distance'] < dis)]
        sum_devices = df_temp['to_pop'].sum()
        return sum_devices

    def Estimate(self,model_which):
        '''
        :param model_which: model
        :return: estimated flow
        '''
        if model_which=='Gravity_model':
            ####Gravity model need parameter estimation
            X = self.df_flow[['from_pop', 'to_pop', 'distance']].values
            Y = self.df_flow['flow'].values

            popt, pcov = curve_fit(self.Gravity_Model, X, Y)
            K, alpha, beta, gamma = popt

            #####Estimation######
            self.df_flow['flow_syn_gravity'] = self.Gravity_Model(X, K, alpha, beta,gamma)

        if model_which=='Radiation_model':

            df_outflow = self.df_flow.groupby(['from_label'])['flow'].sum().reset_index()
            outflow_dict = dict(zip(df_outflow['from_label'], df_outflow['flow']))
            self.df_flow['outflow'] = list(map(lambda x: outflow_dict[x], self.df_flow['from_label']))

            self.df_flow['near_pop'] = list(
                map(lambda x: self.compute_near_pop(x[0],x[1]),
                    zip(self.df_flow['from_label'],  self.df_flow['distance'])))

            #####Estimation######
            X = self.df_flow[['from_pop', 'to_pop', 'outflow','near_pop']].values

            self.df_flow['flow_syn_radiation'] = self.Radiation_Model(X)

        if model_which=='Visitation_model':
            self.df_flow['area'] = self.df_flow['distance']*self.df_flow['distance']*3.14

            self.df_flow['u'] = self.df_flow['unique_usrs']/self.df_flow['area']
            self.df_flow['u']=self.df_flow['u']*(self.df_flow['distance'] *self.df_flow['distance'])

            self.df_flow['f'] = list(map(lambda x: (x[0]+1)/ (x[1]), zip(self.df_flow['max_f'], self.df_flow['min_f'])))

            #####Estimation######
            X = self.df_flow[['u', 'area',  'distance','f']].values

            self.df_flow['flow_syn_visitation'] = self.Visitation_Model(X)

        return self.df_flow














