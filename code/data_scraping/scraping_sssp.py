#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 10:25:34 2021

@author: philipp
"""

import requests, json
import numpy as np
import pandas as pd

# =============================================================================
# CONSTANTS
# =============================================================================
base_url = "https://www.materialscloud.org/mcloud/api/v2/discover/sssp/elements/"

def extract_dE_values(raw_data, potential_key='GBRV_1.2'):
    values = []
    nodes_list = raw_data['data']['Y_series']['delta'][potential_key]['8.0']['points']
    for k in range(len(nodes_list)):
        node = nodes_list[k]
        values.append(node['value'])
    
    return np.array(values)


def extract_Ecutoff_values(raw_data):
    return np.array(raw_data['data']['X_series']['Ecutoff']['values'])


def extract_element_data(element):
    
    # LOADING THE DATA INTO JSON
    url = requests.get(base_url + element)
    text = url.text
    raw_data = json.loads(text)
    
    # EXTRACTING THE VALUES OF INTEREST
    dEs = extract_dE_values(raw_data)
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
    
    test = extract_element_data("H")
    print(test)
    
    element_dfs = []
    for element in periodic_table_keys:
        element_dfs.append(extract_element_data(element))
        # try:
        #     element_dfs.append(extract_element_data(element))
            
        # except:
        #     print("*EXTRACTION ERROR*")
        #     print(element)
            
    df = pd.concat(element_dfs)
            
    
    
    
    
    
    
    





