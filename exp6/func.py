import sys 
sys.path.append("..") 
from utils import *

def extract_ellipsis_item_from_single(r):
    item_lst =[]
    if r[1] == 'fail to follow':
        return []
    orig = r[0].split(' ')
    orig = list(filter(lambda x: x and x.strip(), orig))
    orig = ' '.join(orig).lower().split(' ')
    n_op, ops = minDeletionops(orig, r[1].lower().split(' '))
    
    for idx, op in enumerate(ops):
        if idx == 0:
            cons = [op]
        
        else:
            if op[0] - ops[idx-1][0] != 1:
                item_lst.append(cons)
                cons = [op]
            else:
                cons.append(op)
        if idx == len(ops) - 1:
            item_lst.append(cons)
    return item_lst