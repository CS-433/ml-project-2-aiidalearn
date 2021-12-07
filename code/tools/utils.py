import json
import os
import re
from collections import defaultdict
from typing import Dict

from natsort import natsorted


def load_json(filepath: str):
    with open(filepath) as file:
        data = json.load(file)
    return data


PERIODIC_TABLE_INFO = load_json(
    os.path.join(os.path.dirname(__file__), "periodic_table_info.json",)
)
PTC_COLNAMES = natsorted(
    list(set(PERIODIC_TABLE_INFO[elt]["PTC"] for elt in PERIODIC_TABLE_INFO))
)


def extract_structure_elements(structure_name: str) -> Dict[str, int]:
    """
    Extracts the structure elements from the structure name.
    """
    elts = re.findall("[A-Z][^A-Z]*", structure_name)
    elements_nbrs = defaultdict(int)
    for elt in elts:
        atom_num = re.findall(r"\d+|\D+", elt)
        if len(atom_num) == 1:
            elements_nbrs[elt] += 1
        else:
            elements_nbrs[elt[0]] += int(elt[1])
    return dict(elements_nbrs)
