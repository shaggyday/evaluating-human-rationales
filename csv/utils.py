import pandas as pd
import os
from tqdm import tqdm


def avg(l): return sum(l)/len(l)

'''
converts rationales to binary masks over text
1 to include a token, 0 to exclude
'''
def evidence_to_mask(tokens, evidence_list):
    mask = [0]*len(tokens)
    if evidence_list:
        for evidence in evidence_list:
            if type(evidence) is not dict:
                print("?????")
                print("evidence_to_mask: line 18")
                quit()
            else:
                start_token = evidence['start_token']
                end_token = evidence['end_token']
                for i in range(start_token, end_token):
                    mask[i] = 1
    return mask  