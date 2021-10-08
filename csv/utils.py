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

def text_len_scatter(train, test, val):
    all_texts = list(train['text']) + list(test['text']) + list(val['text'])
    all_text_lens = [len(x.split()) for x in all_texts]
    count_dict = {}
    for text_len in all_text_lens:
        if text_len not in count_dict.keys():
            count_dict[text_len] = 1
        else:
            count_dict[text_len] += 1

    return count_dict