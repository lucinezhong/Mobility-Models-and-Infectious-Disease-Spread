import sys
sys.path.append('/Users/luzhong/Documents/pythonCode/20220101Mobility_Project/')
from input_library import *
import collective_model_fig
import utils

import warnings
warnings.filterwarnings("ignore")

def visitation_law(rf,u):
    return u/np.power(rf,2)

def u_attractiveness_learn(df):
    pi=3.1415926
    base=10
    df=df[df['month']==1]
    output_mat=[]
    for id_index, df_temp in df.groupby('label'):
        df_temp['unique_users/area'] = df_temp['unique_users'] / (df_temp['d_home'] * 2 * pi)
        df_temp['rf'] = df_temp['d_home'] * df_temp['frequency']
        df_temp['rf'] = list(map(lambda x: math.pow(base, int(math.log(x) / math.log(base) / 0.1 + 1) * 0.1), df_temp['rf']))
        df_temp = df_temp.groupby(['rf'])['unique_users/area'].mean().reset_index()
        popt, pov= curve_fit(visitation_law, df_temp['rf'],df_temp['unique_users/area'] )
        u=popt[0]
        print(id_index,u)
        output_mat.append([id_index,u])

    df_u=pd.DataFrame(np.array(output_mat),columns=['county','u'])

    return df_u

if __name__ == "__main__":

    path_data = '/Volumes/SeagateDrive/Mobility_Project/stoppoints_analyze_results/'

    case = 'u_attractiveness'
    if case=='u_attractiveness':
        df = pd.read_csv(path_data + 'F_D_visitation_law_resolutionsum.csv')
        df_u=u_attractiveness_learn(df)
        df_u.to_csv('/Users/lucinezhong/Documents/LuZHONGResearch/20210328Scale_Mobility/Results_visitation/learned_resolution_u.csv')

