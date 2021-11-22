#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 10:25:34 2021

@author: philipp
"""

import requests, json
import numpy as np
import pandas as pd
import os

# =============================================================================
# CONSTANTS
# =============================================================================
base_url = "https://www.materialscloud.org/mcloud/api/v2/discover/sssp/elements/"

# ORIGINAL DICT
# sssp_dict = {
#     '100US': 'RE_pslib.1.0.0_PBE_US_plus_nitrogen',
#     '100PAW': 'RE_pslib.1.0.0_PBE_PAW_plus_nitrogen',
#     '031US': 'pslib.0.3.1_PBE_US',
#     '031PAW': 'pslib.0.3.1_PBE_PAW',
#     'GBRV-1.2': 'GBRV_1.2',
#     'GBRV-1.4': 'GBRV_1.4',
#     'GBRV-1.5': 'GBRV_1.5',
#     'SG15': 'SG15',
#     'SG15-1.1': 'SG15_1.1',
#     'Goedecker': 'Goedecker',
#     'THEOS': 'THEOS',
#     'Dojo': 'Dojo',
#     'Wentzcovitch': 'RE_Wentz_plus_nitrogen'}

# CORRECTED DICT
sssp_dict = {
    '100US': 'pslib.1.0.0_PBE_US',
    '100PAW': 'pslib.1.0.0_PBE_PAW',
    '031US': 'pslib.0.3.1_PBE_US',
    '031PAW': 'pslib.0.3.1_PBE_PAW',
    'GBRV-1.2': 'GBRV_1.2',
    'GBRV-1.4': 'GBRV_1.4',
    'GBRV-1.5': 'GBRV_1.5',
    'SG15': 'SG15',
    'SG15-1.1': 'SG15_1.1',
    'Goedecker': 'Goedecker',
    'THEOS': 'THEOS',
    'Dojo': 'Dojo',
    'Wentzcovitch': 'RE_Wentz_plus_nitrogen'}




def extract_dE_values(raw_data, potential_key='GBRV_1.2'):
    values = []
    try:
        nodes_list = raw_data['data']['Y_series']['delta'][potential_key]['4.0']['points']
    except:
        nodes_list = raw_data['data']['Y_series']['delta'][potential_key]['8.0']['points']
    for k in range(len(nodes_list)):
        node = nodes_list[k]
        if node == None: #TODO: Choose better dummy value
            values.append(0)
        else:
            values.append(node['value'])
    
    return np.array(values)


def extract_Ecutoff_values(raw_data):
    return np.array(raw_data['data']['X_series']['Ecutoff']['values'])


def extract_element_data(element, potential_key):
    
    # LOADING THE DATA INTO JSON
    url = requests.get(base_url + element)
    text = url.text
    raw_data = json.loads(text)
    
    # EXTRACTING THE VALUES OF INTEREST
    dEs = extract_dE_values(raw_data, potential_key)
    Ecutoffs = extract_Ecutoff_values(raw_data)
    
    # INITIALIZE ELEMENT COLUMN
    elements = [element for k in range(len(dEs))]
    
    # CONSTRUCT DATAFRAME
    df_element = pd.DataFrame(
            {'element' : elements,
             'ecutoff' : Ecutoffs,
             'dE' : dEs
                })
    return df_element

if __name__ == '__main__':
    
    # LOADING ALL ELEMENT KEYS
    url_table = requests.get("https://archive.materialscloud.org/record/file?record_id=862&filename=SSSP_1.1.2_PBE_efficiency.json&file_id=a5642f40-74af-4073-8dfd-706d2c7fccc2")
    text_table = url_table.text
    sssp_table = json.loads(text_table)
    periodic_table_keys = list(sssp_table.keys())
    remove_elements = ['Ce',
                       'Dy',
                       'Er',
                       'Eu',
                       'F',
                       'Gd',
                       'Ho',
                       'La',
                       'Lu',
                       'N', #Potential not found
                       'Nd',
                       'Pm',
                       'Pr',
                       'Sm',
                       'Tb',
                       'Tm',
                       'Yb'
                       ]
    
    for element in remove_elements:
        periodic_table_keys.remove(element)
 
    element_dfs = []
    for element in periodic_table_keys:
        print("-----", element, "-----")
        potential_shortname = sssp_table[element]['pseudopotential']
        # print(potential_shortname)
        potential_key = sssp_dict[potential_shortname]
        element_dfs.append(extract_element_data(element, potential_key))
        # try:
        #     element_dfs.append(extract_element_data(element))
            
        # except:
        #     print("*EXTRACTION ERROR*")
        #     print(element)
            
    df = pd.concat(element_dfs)
            
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data/sssp_data.csv")
    
    df.to_csv(DATA_PATH)
    print("-----FINISHED PARSING-----")
    print("ISSUES WITH ", len(remove_elements), " ELEMENTS")
    print("--------------------------")
    
    
    
    





