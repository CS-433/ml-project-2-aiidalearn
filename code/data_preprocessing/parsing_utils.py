#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:32:26 2021

@author: philipp
"""

import pandas as pd
import numpy as np
import json
import os
import requests

# LOADING ALL ELEMENT KEYS
url_table = requests.get("https://archive.materialscloud.org/record/file?record_id=862&filename=SSSP_1.1.2_PBE_efficiency.json&file_id=a5642f40-74af-4073-8dfd-706d2c7fccc2")
text_table = url_table.text
sssp_table = json.loads(text_table)
periodic_table_keys = list(sssp_table.keys())

def encode_structure(df, elements_nbrs):
    
    total_atoms = sum(list(elements_nbrs.values()))
    
    elements = list(elements_nbrs.keys())
    
    for element in elements:
        df[element] = elements_nbrs[element]/total_atoms
    
    return df

def parse_json(filepath, savepath, elements_nbrs):
    
    with open(filepath) as file:
        data = json.load(file)
        
    raw_df = pd.DataFrame(data)
    rel_cols = ['ecutrho', 'k_density', 'ecutwfc', 'converged' , 'accuracy', 'total_energy']
    df = raw_df[rel_cols]
    
    for element in periodic_table_keys:
        df[element] = 0.0
        
    df = encode_structure(df, elements_nbrs)
    
    df.to_csv(savepath)


if __name__ == "__main__":
    elements_nbrs = {
        "Ge" : 1,
        "Te" : 1
        }
    
    filepath = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data/data.json")
    savepath = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data/toy_data2.csv")
    
    parse_json(filepath, savepath, elements_nbrs)
    
    




