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

def class_distr(df):
    classifications = list(df['classification'])
    class_dict = {}
    for c in classifications:
        if c not in class_dict.keys():
            class_dict[c] = 1
        else:
            class_dict[c] += 1

    counts = list(class_dict.values())
    distribution = [x/sum(counts) for x in counts]

    return class_dict, distribution