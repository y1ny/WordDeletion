import random
from nltk import Tree
from functools import reduce
import sys 
sys.path.append("..") 
from utils import *
from scripts import process_ptb

class EllipsisTreeNode(object):
    """The basic node of tree structure"""

    def __init__(self, name, value=0, parent=None):
        super(EllipsisTreeNode, self).__init__()
        self.name = name
        self.parent = parent
        self.child = {}
        self.value = value

    def __repr__(self) :
        return 'EllipsisTreeNode(%s)' % self.name

    def __contains__(self, item):
        return item in self.child


    def __len__(self):
        """return number of children node"""
        return len(self.child)
    
    def __eq__(self, other):
        if isinstance(other, EllipsisTreeNode):
            return self.name == other.name
        return NotImplemented

    @property
    def path(self):
        """return path string (from root to current node)"""
        if self.parent:
            return f"{self.parent.path.strip()}->{self.name}"
        else:
            return self.name

    def get_subtree(self):
        if not self.child:
            return {}
        child_dict = {node: {} for node in self.child.keys()}
        for name, node in self.child.items():
            child_dict[name] = node.get_subtree()
        return child_dict
    
    def get_subtree_value(self):
        value_count = self.value
        if self.child:
            for name, node in self.child.items():
                value_count += node.get_subtree_value()
            return value_count
        else:
            return value_count
        

    def get_child(self, name, defval=None):
        """get a child node of current node"""
        return self.child.get(name, defval)

    def add_child(self, name, obj):
        """add a child node to current node"""
        if obj and not isinstance(obj, EllipsisTreeNode):
            raise ValueError('TreeNode only add another TreeNode obj as child')
        if obj is None:
            obj = EllipsisTreeNode(name)
        obj.parent = self
        self.child[name] = obj
        return obj

    def del_child(self, name):
        """remove a child node from current node"""
        if name in self.child:
            del self.child[name]

    def find_child(self, path, create=False):
        """find child node by path/name, return None if not found"""
        # convert path to a list if input is a string
        path = path if isinstance(path, list) else path.split()
        cur = self
        for sub in path:
            # search
            obj = cur.get_child(sub)
            if obj is None and create:
                # create new node if need
                obj = cur.add_child(sub)
            # check if search done
            if obj is None:
                break
            cur = obj
        return obj

    def items(self):
        return self.child.items()

    def dump(self, indent=0):
        """dump tree to string"""
        tab = '    '*(indent-1) + ' |- ' if indent > 0 else ''
        print('%s%s' % (tab, self.name))
        for name, obj in self.items():
            obj.dump(indent+1)
                
    
    def add_parent(self, obj):
        if obj and not isinstance(obj, EllipsisTreeNode):
            raise ValueError('TreeNode only add another TreeNode obj as child')
        self.parent = obj
        
    def node_distance(self, node):
        set1 = set(self.name.split('-'))
        set2 = set(node.name.split('-'))
        if not (set1.issubset(set2) or set2.issubset(set1)):
            return None
        if set1.issubset(set2):
            return - len(set2 - set1)
        else:
            return len(set1 - set2)

def binarize(tree):
    """
    Recursively turn a tree into a binary tree.
    """
    if isinstance(tree, str):
        return tree
    elif len(tree) == 1:
        return binarize(tree[0])
    else:
        label = tree.label()
        return reduce(lambda x, y: Tree(label, (binarize(x), binarize(y))), tree)
    
def random_xp_retriever(target, cons_feature_dict):
    candidate_node_dict = {}
    for sent in cons_feature_dict.keys():
        if sent.lower() == target.lower():
            continue
        if len(cons_feature_dict[sent][1]) not in range(3,9):
            continue
        for node in cons_feature_dict[sent][-2].keys():
            if node.split('-')[0] not in process_ptb.phrase_label + process_ptb.sent_label:
                continue
            if len(cons_feature_dict[sent][2][node]) == len(sent.split(' ')):
                continue
            parent = cons_feature_dict[sent][-2][node][-1]
            node_key = f"{parent.split('-')[0]}-{node.split('-')[0]}"
            if node_key not in candidate_node_dict.keys():
                candidate_node_dict[node_key] = []
            candidate_node_dict[node_key].append([sent, node])
    
    sent_lst = []
    for node_key in candidate_node_dict.keys():
        tmp = candidate_node_dict[node_key]
        if len(tmp) < 10:
            sent_lst.extend(tmp)
        else:
            sent_lst.extend(random.sample(tmp, 10))
    
    return sent_lst

def construct_retriever_set(target, output_path, cons_feature_dict):
    sent_lst = random_xp_retriever(target, cons_feature_dict)
    with open(f'{output_path}.csv', 'w') as f:
        f.write("demonstration\tdemon_label\tnode\tparent\tsentence\tlabel\n")
        for item in sent_lst:
            demon = item[0]
            demon_label_cons = item[1]
            node_idx = [item[0] for item in cons_feature_dict[demon][2][demon_label_cons]]
            tmp = []
            for idx, node in enumerate(demon.split(' ')):
                if idx in node_idx:
                    continue
                tmp.append(node)
            demon_label = ' '.join(tmp)
            parent_type = cons_feature_dict[demon][-2][demon_label_cons][-1]
            f.write("\t".join([demon, demon_label, demon_label_cons, parent_type, target, target]) + '\n')
    return


def extract_ellipsis_item(result):
    item_lst = []
    idx_dict = {}
    for idx, r in enumerate(result):
        if r[2] == 'fail to follow':
            continue
        orig = r[0].split(' ')
        orig = list(filter(lambda x: x and x.strip(), orig))
        orig = ' '.join(orig).lower().split(' ')
        n_op, ops = minDeletionops(orig, r[2].lower().split(' '))
        
        for idx, op in enumerate(ops):
            if idx == 0:
                cons = [op]
            
            else:
                if op[0] - ops[idx-1][0] != 1:
                    idx_lst = [str(item[0]) for item in cons]
                    if '-'.join(idx_lst) not in idx_dict.keys():
                        idx_dict['-'.join(idx_lst)] = 0
                    idx_dict['-'.join(idx_lst)] += 1
                    item_lst.append(cons)
                    cons = [op]
                else:
                    cons.append(op)
            if idx == len(ops) - 1:
                idx_lst = [str(item[0]) for item in cons]
                if '-'.join(idx_lst) not in idx_dict.keys():
                    idx_dict['-'.join(idx_lst)] = 0
                idx_dict['-'.join(idx_lst)] += 1
                item_lst.append(cons)
    return item_lst, idx_dict

def extract_ellipsis_item_ctb(result):
    item_lst = []
    idx_dict = {}
    for idx, r in enumerate(result):
        if r[2] == 'fail to follow':
            continue
        orig = list(r[0].strip())
        n_op, ops = minDeletionops(orig, list(r[2].strip()))
                
        for idx, op in enumerate(ops):
            if idx == 0:
                cons = [op]
            
            else:
                if op[0] - ops[idx-1][0] != 1:
                    idx_lst = [str(item[0]) for item in cons]
                    if '-'.join(idx_lst) not in idx_dict.keys():
                        idx_dict['-'.join(idx_lst)] = 0
                    idx_dict['-'.join(idx_lst)] += 1
                    item_lst.append(cons)
                    cons = [op]
                else:
                    cons.append(op)
            if idx == len(ops) - 1:
                idx_lst = [str(item[0]) for item in cons]
                if '-'.join(idx_lst) not in idx_dict.keys():
                    idx_dict['-'.join(idx_lst)] = 0
                idx_dict['-'.join(idx_lst)] += 1
                item_lst.append(cons)
    return item_lst, idx_dict


def get_name(i, j):
    lst = range(i, j)
    lst = list(map(str, lst))
    return '-'.join(lst)

def backpt_to_tree(sent, backpt, span_prob):
    """
    backpt[i][j] - the best splitpoint for the span sent[i:j]
    label_table[i][j] - description for span sent[i:j] (for humans to read - the parser is unlabeled)
    """
    root = EllipsisTreeNode(get_name(0, len(sent)), 1)
    def to_tree(i, j, parent):
        if j - i == 1:
            return EllipsisTreeNode(str(i), 1, parent)
        else:
            k = backpt[i][j]
            name = get_name(i, j)
            if name not in span_prob.keys():
                prob = 0
            else:
                prob = span_prob[name]
            node = EllipsisTreeNode(name, prob, parent)
            node.add_child(get_name(i, k),to_tree(i,k, node))
            node.add_child(get_name(k, j),to_tree(k,j, node))
            return node
    return to_tree(0, len(sent), root)

def compute_score(span_lst, score_dict):
    score = 0 
    for span in span_lst:
        if span not in score_dict.keys():
            continue
        score += score_dict[span]
    return score

def is_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def cky(target, idx_dict, method='add'):
    '''
    method can be 'add', 'multiply', 'add-log', 'multiply-log'
    '''
    if is_contain_chinese(target):
        target = list(target)
    else:
        target = target.split(' ')
    idx_max = 0
    idx_sum = 0
    for k, v in idx_dict.items():
        if len(k.split('-')) == 1:
            continue
        idx_sum += v
        if v > idx_max:
            idx_max = v
    # construct probability table based on the output span dict
    if 'log' in method:
        prob_table = [[np.log(1e-20) for _ in range(len(target) + 1)] for _ in range(len(target))]
    else:
        prob_table = [[0 for _ in range(len(target) + 1)] for _ in range(len(target))]
    span_prob = {}
    for span, value in idx_dict.items():
        tmp = span
        span = span.split('-')
        span = list(map(int, span))
        if len(span) == 1:
            continue
        probs = value / idx_max
        if 'log' in method:
            prob_table[span[0]][span[-1]+1] = np.log(probs)
            span_prob[tmp] = np.log(value / idx_sum)
        else:
            prob_table[span[0]][span[-1]+1] = probs
            span_prob[tmp] = value / idx_sum
    
    # perform cky using scores and backpointers
    score_table = [[None for _ in range(len(target) + 1)] for _ in range(len(target))]
    backpt_table = [[None for _ in range(len(target) + 1)] for _ in range(len(target))]
    for i in range(len(target)): 
        if 'log' in method:
            if 'multiply' in method:
                score_table[i][i+1] = np.log(0.999999)
            else:
                score_table[i][i+1] = np.log(1)
        else:
            score_table[i][i+1] = 1
    for j in range(2, len(target) + 1): 
        for i in range(j-2, -1, -1):
            best, argmax = -np.inf, None
            for k in range(i+1, j): # find splitpoint
                if 'add' in method:
                    score = score_table[i][k] + score_table[k][j]
                elif 'multiply' in method:
                    if 'log' in method:
                        score = -abs(score_table[i][k] * score_table[k][j])
                    else:
                        score = score_table[i][k] * score_table[k][j]
                
                if score > best:
                    best, argmax = score, k
            score_table[i][j] = best + prob_table[i][j]
            backpt_table[i][j] = argmax 
    tree = backpt_to_tree(target, backpt_table, span_prob)
    return tree

from treelib import Node, Tree
def display_tree(root, target):
    from treelib import Node, Tree
    target = target.split(' ')
    tree = Tree()
    node_lst = [root]
    class NodeInfo(object):
        def __init__(self, name, value):
            self.content = f"{name}-{value:.4f}"
    while node_lst:
        tmp = []
        for node in node_lst:
            word = []
            for idx in node.name.split('-'):
                word.append(target[int(idx)])
            word = ' '.join(word)
            if node.parent:
                tree.create_node(word, node.name, node.parent.name, data=NodeInfo(word, node.value))
            else:
                tree.create_node(word, node.name, data=NodeInfo(word, node.value))
            tmp.extend([[key, item] for key, item in node.child.items()])
        tmp = list(tmp)
        tmp = sorted(tmp, key = lambda x: int(x[0].split('-')[0]))
        node_lst = [item[1] for item in tmp]
    tree.show(key=False, data_property='content')
    return tree

def tree_to_bracket(node, target_lst):

    if not node.child:
        word_idx = node.name
        word_idx = word_idx.split('-')
        word_idx = list(map(int, word_idx))
        word_lst = [target_lst[idx] for idx in word_idx]
        word_lst = '-'.join(word_lst)
        return "{" + word_lst + "}"
    else:
        children_str = ''.join(tree_to_bracket(child, target_lst) for name, child in node.child.items())
        
        word_idx = node.name
        word_idx = word_idx.split('-')
        word_idx = list(map(int, word_idx))
        word_lst = [target_lst[idx] for idx in word_idx]
        word_lst = '-'.join(word_lst)
        return "{" + word_lst + children_str + "}"

def tree_to_span(tree:EllipsisTreeNode):
    span_lst = []
    node_lst = [tree]
    root = tree.name
    while node_lst:
        tmp = []
        for node in node_lst:
            if len(node.name.split('-')) != 1 and node.name != root:
                span_lst.append(node.name)
            tmp.extend([[key, item] for key, item in node.child.items()])
        tmp = list(tmp)
        tmp = sorted(tmp, key = lambda x: int(x[0].split('-')[0]))
        node_lst = [item[1] for item in tmp]
    return list(set(span_lst))

def count_nodes(root):
    if not root:
        return 1
    child_lst = list(root.child.keys())
    child_lst = sorted(child_lst, key = lambda x: int(x.split('-')[0]))
    return 1 + count_nodes(root.child[child_lst[0]]) + count_nodes(root.child[child_lst[1]])

def measure_left_right_size(parsed_lst):
    left_lst = []
    right_lst = []
    ratio_lst = []
    for idx, parsed in enumerate(parsed_lst):
        node_lst = [parsed]
        left = 0
        right = 0
        while True:
            tmp = []
            for node in node_lst:
                child_lst = list(node.child.keys())
                child_lst = sorted(child_lst, key = lambda x: int(x.split('-')[0]))
                if not child_lst:
                    continue
                left += count_nodes(node.child[child_lst[0]])
                right += count_nodes(node.child[child_lst[1]])
                tmp.extend([v for _, v in node.child.items()])
            node_lst = tmp
            if tmp == []:
                break
        left_lst.append(left )
        right_lst.append(right )
        ratio_lst.append(right/left)
    return left_lst, right_lst, ratio_lst