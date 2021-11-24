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

# ULTRASOFT POTENTIALS
US = [
      'pslib.1.0.0_PBE_US',
      'pslib.1.0.0_PBE_PAW',
      'pslib.0.3.1_PBE_US',
      'pslib.0.3.1_PBE_PAW',
      'GBRV_1.2',
      'GBRV_1.4',
      'GBRV_1.5',
      ]

# NORMCONSERVING POTENTIALS
NC = [
      'SG15',
      'SG15_1.1',
      'Dojo'
      ]
# UNCLASSIFIED POTENTIALS
OTHER = [
    'Goedecker',
    'THEOS',
    'RE_Wentz_plus_nitrogen'
    ]

 


def get_dual(potential_key, element):
    
    if element == ('Fe' or 'Mn'):
        return 12.0
    
    
    elif potential_key in US:
        return 8.0
    
    elif potential_key in NC:
        return 4.0
    
    else:
        return 4.0 #TODO: Check for correct dual


def extract_values(raw_data, potential_key='GBRV_1.2'):
    E_val = []
    missing = 0
    
    
    try:
        nodes_list = raw_data['data']['Y_series']['cohesive_energy'][potential_key]['4.0']['points']
    except:
        nodes_list = raw_data['data']['Y_series']['cohesive_energy'][potential_key]['8.0']['points']
        
    for k in range(len(nodes_list)):
        node = nodes_list[k]
        # TODO: subtract cohesive_energy with largest cutoff
        if node == None:
            E_val.append(-1)
            missing += 1
        else:
            E_val.append(node['value'])
    
    Ecutoff_val = raw_data['data']['X_series']['Ecutoff']['values']
    max_cutoff = max(Ecutoff_val)
    max_idx = Ecutoff_val.index(max_cutoff)
    dE_val = np.array(E_val) - E_val[max_idx]

    
    return dE_val, np.array(Ecutoff_val), missing



def extract_element_data(element, potential_key):
    
    # LOADING THE DATA INTO JSON
    url = requests.get(base_url + element)
    text = url.text
    raw_data = json.loads(text)
    
    # EXTRACTING THE VALUES OF INTEREST
    element_missing = 0
    dEs, Ecutoffs, missing = extract_values(raw_data, potential_key)
    element_missing += missing
    dual = get_dual(potential_key, element)
    
    # INITIALIZE ELEMENT COLUMN
    elements = [element for k in range(len(dEs))]
    
    # CONSTRUCT DATAFRAME
    df_element = pd.DataFrame(
            {'element' : elements,
             'ecutwfc' : Ecutoffs,
             'ecutrho' : dual*Ecutoffs,
             'dE' : dEs
                })
    return df_element, element_missing

if __name__ == '__main__':
    
    # LOADING ALL ELEMENT KEYS
    url_table = requests.get("https://archive.materialscloud.org/record/file?record_id=862&filename=SSSP_1.1.2_PBE_efficiency.json&file_id=a5642f40-74af-4073-8dfd-706d2c7fccc2")
    text_table = url_table.text
    sssp_table = json.loads(text_table)
    periodic_table_keys = list(sssp_table.keys())
    remove_elements = [
                       'F' , #Potential not found
                       'N' , #Potential not found
                       ]
    
    for element in remove_elements:
        periodic_table_keys.remove(element)
 
    element_dfs = []
    elements_w_missing = []
    global_missing = 0
    for element in periodic_table_keys:
        print("-----", element, "-----")
        potential_shortname = sssp_table[element]['pseudopotential']
        potential_key = sssp_dict[potential_shortname]
        element_df, element_missing = extract_element_data(element, potential_key)
        element_dfs.append(element_df)
        if element_missing:
            elements_w_missing.append(element)
        
        
        global_missing += element_missing
        
        print("PSEUDO: ", potential_key)
        print(element_missing, " MISSING VALUES")
        print("--------------")

    df = pd.concat(element_dfs)
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data/sssp_data.csv")
    
    df.to_csv(DATA_PATH)
    print("-----FINISHED PARSING-----")
    print("* ISSUES WITH ", len(remove_elements), " ELEMENTS:")
    print(remove_elements)
    print("* ", global_missing, " MISSING VALUES")
    print("* ELEMENTS WITH MISSING VALUES: ", elements_w_missing)
    print("--------------------------")
    
    
    
    





