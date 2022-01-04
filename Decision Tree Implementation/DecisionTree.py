import numpy as np
import os
import graphviz
import math

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    
    # Partition x into two sets containing instances where 0 occur and instances where 1 occur 
    partitionSet = {}
    values, count = np.unique(x, return_counts = "true")
    for v in values: # List comprehension iteration
        partitionSet[v] = [i for i, k in enumerate(x) if k == v] # All places of i where v is present
       
    return partitionSet
    raise Exception('Partition Function Exception')

def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z
    
    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    
    h = 0
    values, count = np.unique(y, return_counts = "true")   
    for c in count:
        h = h + ((c/len(y)) * math.log((c/len(y)),2))   
    h = -1 * h   
    
    return h
    raise Exception("Entropy Function Exception")

def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    
    MI = {}
    values, count = np.unique(x, return_counts = "true")
    y_entropy = entropy(y)
    for v in values:
        newYa = y[np.where(x == v)]
        newYb = y[np.where(x != v)]
        MI[v] = y_entropy - (((len(newYa)/len(y)) * entropy(newYa)) + ((len(newYb)/len(y)) * entropy(newYb)))   
    first_key = list(MI.keys())[0]   # segment initial value for single number comparison
    
    return MI[first_key]   
    raise Exception("MI Function Exception")

def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The splitting criterion has to be chosen from among all possible attribute-value pairs. That is, for a problem with two 
    features/attributes x1 (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is 
    a list of all pairs of attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    
    # Creating the attribute value pairs for binary format classification
    if attribute_value_pairs == None:
        attribute_value_pairs = []
        x_transpose = x.transpose()
        for i in range(0,np.size(x[0])):
            for k in x_transpose[i]:
                if (i, k) not in attribute_value_pairs:
                    attribute_value_pairs.append((i, k))

    # Base Condition 1 for termination
    if len(y) == 0:
        return None        
    if (len(set(y)) == 1):
        return y[0]
    # Base Condition 2 & 3 for termination
    if len(attribute_value_pairs) == 0 or depth == max_depth:
        values, count = np.unique(y, return_counts = "true")
        return values[np.argmax(count)] # returns majority label
    
    # Initialization of variables needed for finding optimal pair, setting node, and splitting tree.
    x_column = x.transpose()
    optimalPair = ()
    optimalCol = []
    nestedDT = {}
    xOptionA = x
    xOptionB = x
    yOptionA = y
    yOptionB = y
    weight1 = 0
    weight2 = 0
    mostMI = -1
    
    # Optimal Pair & Column
    for attributePair in attribute_value_pairs:
        xiCol = x_column[attributePair[0]]
        tempMI = mutual_information(xiCol, y)
        if tempMI > mostMI:
            mostMI = tempMI
            optimalPair = attributePair
            optimalCol = xiCol
            
    attribute_value_pairs.remove(optimalPair)
            
    # Initial Node & Tree Split        
    for i in range(0,len(x)):
        if optimalCol[i] != optimalPair[1]:
            xOptionA = np.delete(xOptionA, weight1, 0)
            yOptionA = np.delete(yOptionA, weight1, 0)
            weight2 += 1
        else:
            xOptionB = np.delete(xOptionB, weight2, 0)
            yOptionB = np.delete(yOptionB, weight2, 0)
            weight1 += 1
    
    # ID3 recursion for when attribute pair is true & false
    nestedDT[(optimalPair[0], optimalPair[1], True)] = id3(xOptionA, yOptionA, attribute_value_pairs, depth + 1, max_depth)
    nestedDT[(optimalPair[0], optimalPair[1], False)] = id3(xOptionB, yOptionB, attribute_value_pairs, depth + 1, max_depth)
    
    return nestedDT      
    raise Exception("ID3 Function Exception")

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    
    # Each 6 attributes of a value is checked against the tree. Iteration occurs by
    # comparing the tree key values with the current x value until a final label of 0 or 1 is reached.
    for i in tree.keys():
        if(i[1] == x[i[0]] and i[2] == True):
            if(tree[i] != 0 and tree[i] != 1):
                return predict_example(x, tree[i])
            else:
                return tree[i]
        if(i[1] != x[i[0]] and i[2] == False):
            if(tree[i] != 0 and tree[i] != 1):
                return predict_example(x, tree[i])
            else:
                return tree[i]
            
    raise Exception("PredictExample Function Exception")
    
def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    
    # Calculates error sum for number of times y_true and y_pred differ. Return a percentage, 0: best & 1: worst.
    errorSum = 0;
    for i in range(0,len(y_true)):
        if(y_true[i] != y_pred[i]):
            errorSum += 1
    errorSum = (errorSum/len(y_true))
    
    return errorSum
    raise Exception("ComputeError Function Exception")

def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console: the raw nested dictionary representation
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)

def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid

if __name__ == '__main__':
    
    # Load the training data
    M = np.genfromtxt('./monks_data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks_data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=5)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    #dot_str = to_graphviz(decision_tree)
    #render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    #y_pred = [predict_example(x, decision_tree) for x in Xtst]
    #tst_err = compute_error(ytst, y_pred)
    #print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
