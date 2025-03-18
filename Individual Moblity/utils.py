
import h3
from geopy.geocoders import Nominatim
import reverse_geocoder as rg

import math
import numpy as np
def cells_to_h3(pos,resolution):
    new_label = h3.geo_to_h3(pos[0], pos[1], resolution=resolution)
    return new_label


def cells_to_county(pos):
    coordinates = pos
    location = rg.search(coordinates)
    new_label = location[0]['admin1'] + "_" + location[0]['admin2']
    return new_label


def GR_powerlaws(para,min_v,max_v,num): ###generating random variable
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


def GR_random( min_v,max_v,num):###generating random variable
    x = np.random.uniform(min_v, max_v, num)
    return x


def find_move_loc(pos,move_r,move_a):
    '''
    :param pos: current position
    :param move_r: move_r
    :param move_a: move_a
    :return: next position
    '''
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
    return (round(lat2,4),round(lon2,4))

def haversine(points_a, points_b, radians=False):
    """
    Calculate the great-circle distance bewteen points_a and points_b
    points_a and points_b can be a single points or lists of points.

    Author: Piotr Sapiezynski
    Source: https://github.com/sapiezynski/haversinevec

    Using this because it is vectorized (stupid fast).
    """

    def _split_columns(array):
        if array.ndim == 1:
            return array[0], array[1]  # just a single row
        else:
            return array[:, 0], array[:, 1]

    if radians:
        lat1, lon1 = _split_columns(points_a)
        lat2, lon2 = _split_columns(points_b)

    else:
        # convert all latitudes/longitudes from decimal degrees to radians
        lat1, lon1 = _split_columns(np.radians(points_a))
        lat2, lon2 = _split_columns(np.radians(points_b))

    # calculate haversine
    lat = lat2 - lat1
    lon = lon2 - lon1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lon * 0.5) ** 2
    h = 2 * 6371 * np.arcsin(np.sqrt(d))
    return h  # in kilometers

def get_bearing(points_a, points_b):
    (lat1,lon1)=points_a
    (lat2, lon2) = points_b
    dLon = (lon2 - lon1)
    x = np.cos(np.radians(lat2)) * np.sin(np.radians(dLon))
    y = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)

    return brng
