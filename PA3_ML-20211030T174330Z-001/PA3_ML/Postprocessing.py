
#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: #
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""
from collections import defaultdict

from utils import apply_threshold, get_total_accuracy, get_positive_predictive_value, get_num_predicted_positives, \
    get_true_positive_rate, get_num_correct


def enforce_demographic_parity(categorical_results, epsilon):
    demographic_parity_data = {}
    thresholds = {}
    maximum = 0
    import numpy as np
    def compare_probs(prob1, prob2, epsilon):
        return abs(prob1 - prob2) <= epsilon

    threshold = np.arange(0.0, 1.0, 0.01)
    demo = defaultdict(list)
    for i in threshold:
        for j in categorical_results.keys():
            applied_threshold_value = apply_threshold(categorical_results[j], i)
            pp_values = get_num_predicted_positives(applied_threshold_value)/len(applied_threshold_value)
            demo[j].append((pp_values, i))

    list1 = []
    for i in list(demo.values())[0]:
        for j in list(demo.values())[1]:
            if compare_probs(i[0], j[0], epsilon):
                list1.append([i, j])
    list2 = []
    for a in list1:
        for b in list(demo.values())[2]:
            if compare_probs(a[0][0], b[0], epsilon):
                if compare_probs(a[1][0], b[0], epsilon):
                    list2.append([a[0], a[1], b])
    list3 = []
    for s in list2:
        for t in list(demo.values())[3]:
            if compare_probs(s[0][0], t[0], epsilon):
                if compare_probs(s[1][0], t[0], epsilon):
                    if compare_probs(s[2][0], t[0], epsilon):
                        list3.append([s[0], s[1], s[2], t])
    classification = {}

    for i in list3:
        classification[list(categorical_results.keys())[0]] = (
            apply_threshold(list(categorical_results.values())[0], i[0][1]))
        classification[list(categorical_results.keys())[1]] = (
            apply_threshold(list(categorical_results.values())[1], i[1][1]))
        classification[list(categorical_results.keys())[2]] = (
            apply_threshold(list(categorical_results.values())[2], i[2][1]))
        classification[list(categorical_results.keys())[3]] = (
            apply_threshold(list(categorical_results.values())[3], i[3][1]))

        accuracy = get_total_accuracy(classification)
        if accuracy > maximum:
            maximum = accuracy
            thresholds[list(categorical_results.keys())[0]] = (i[0][1])
            thresholds[list(categorical_results.keys())[1]] = (i[1][1])
            thresholds[list(categorical_results.keys())[2]] = (i[2][1])
            thresholds[list(categorical_results.keys())[3]] = (i[3][1])
            demographic_parity_data = classification.copy()

    return demographic_parity_data, thresholds

    # Must complete this function!
    #return demographic_parity_data, thresholds

    #return None, None

#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):
    #get_true_positive_rate
    thresholds = {}
    equal_opportunity_data = {}
    maximum = 0
    import numpy as np
    def compare_probs(prob1, prob2, epsilon):
        return abs(prob1 - prob2) <= epsilon

    thresholds = {}
    threshold = np.arange(0.0, 1.0, 0.01)
    equal = defaultdict(list)
    for i in threshold:
        for j in categorical_results.keys():
            applied_threshold_value = apply_threshold(categorical_results[j], i)
            pp_values = get_true_positive_rate(applied_threshold_value)
            equal[j].append((pp_values, i))

    list1 = []
    for i in list(equal.values())[0]:
        for j in list(equal.values())[1]:
            if compare_probs(i[0], j[0], epsilon):
                list1.append([i, j])
    list2 = []
    for a in list1:
        for b in list(equal.values())[2]:
            if compare_probs(a[0][0], b[0], epsilon):
                if compare_probs(a[1][0], b[0], epsilon):
                    list2.append([a[0], a[1], b])
    list3 = []
    for s in list2:
        for t in list(equal.values())[3]:
            if compare_probs(s[0][0], t[0], epsilon):
                if compare_probs(s[1][0], t[0], epsilon):
                    if compare_probs(s[2][0], t[0], epsilon):
                        list3.append([s[0], s[1], s[2], t])
    classification = {}

    for i in list3:
        classification[list(categorical_results.keys())[0]] = (
            apply_threshold(list(categorical_results.values())[0], i[0][1]))
        classification[list(categorical_results.keys())[1]] = (
            apply_threshold(list(categorical_results.values())[1], i[1][1]))
        classification[list(categorical_results.keys())[2]] = (
            apply_threshold(list(categorical_results.values())[2], i[2][1]))
        classification[list(categorical_results.keys())[3]] = (
            apply_threshold(list(categorical_results.values())[3], i[3][1]))

        accuracy = get_total_accuracy(classification)
        if accuracy > maximum:
            maximum = accuracy
            thresholds[list(categorical_results.keys())[0]] = (i[0][1])
            thresholds[list(categorical_results.keys())[1]] = (i[1][1])
            thresholds[list(categorical_results.keys())[2]] = (i[2][1])
            thresholds[list(categorical_results.keys())[3]] = (i[3][1])
            equal_opportunity_data = classification.copy()

    return equal_opportunity_data, thresholds


    # Must complete this function!
    #return equal_opportunity_data, thresholds

    #return None, None

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    mp_data = {}
    thresholds = {}

    import numpy as np
    threshold = np.arange(0.0, 1.0, 0.01)
    # print(threshold)
    dict_applied_threshold_values = defaultdict(list)
    dict_pp_value = defaultdict(list)

    for i in categorical_results.keys():
        max_accuracy_value = 0
        max_threshold_value = 0
        for j in threshold:
            applied_threshold_value = apply_threshold(categorical_results[i], j)
            accuracy_value = get_num_correct(applied_threshold_value) / len(applied_threshold_value)
            if accuracy_value > max_accuracy_value:
                max_accuracy_value = accuracy_value
                max_threshold_value = j
        dict_applied_threshold_values[i] = max_threshold_value

        dict_pp_value[i] = apply_threshold(categorical_results[i], dict_applied_threshold_values[i])

    # Must complete this function!
    return dict_pp_value, dict_applied_threshold_values
    # Must complete this function!
    #return mp_data, thresholds

    #return None, None

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}
    maximum = 0
    import numpy as np
    def compare_probs(prob1, prob2, epsilon):
        return abs(prob1 - prob2) <= epsilon
    predictive_parity_data = {}
    thresholds = {}
    threshold = np.arange(0.0, 1.0, 0.01)
    ppv = defaultdict(list)
    for i in threshold:
        for j in categorical_results.keys():

            applied_threshold_value = apply_threshold(categorical_results[j], i)
            pp_values = get_positive_predictive_value(applied_threshold_value)
            ppv[j].append((pp_values, i))

    list1=[]
    for i in list(ppv.values())[0]:
        for j in list(ppv.values())[1]:
            if compare_probs(i[0], j[0], epsilon):
                list1.append([i, j])
    list2=[]
    for a in list1:
        for b in list(ppv.values())[2]:
            if compare_probs(a[0][0], b[0], epsilon):
                if compare_probs(a[1][0], b[0], epsilon):
                    list2.append([a[0],a[1], b])
    list3 = []
    for s in list2:
        for t in list(ppv.values())[3]:
            if compare_probs(s[0][0], t[0], epsilon):
               if compare_probs(s[1][0], t[0], epsilon):
                    if compare_probs(s[2][0], t[0], epsilon):
                      list3.append([s[0], s[1], s[2], t])
    classification={}

    for i in list3:
        classification[list(categorical_results.keys())[0]] = (apply_threshold(list(categorical_results.values())[0],i[0][1]))
        classification[list(categorical_results.keys())[1]] = (apply_threshold(list(categorical_results.values())[1], i[1][1]))
        classification[list(categorical_results.keys())[2]] = (apply_threshold(list(categorical_results.values())[2], i[2][1]))
        classification[list(categorical_results.keys())[3]] = (apply_threshold(list(categorical_results.values())[3], i[3][1]))

        accuracy=get_total_accuracy(classification)
        if accuracy>maximum:
            maximum=accuracy
            thresholds[list(categorical_results.keys())[0]] = (i[0][1])
            thresholds[list(categorical_results.keys())[1]] = (i[1][1])
            thresholds[list(categorical_results.keys())[2]] = (i[2][1])
            thresholds[list(categorical_results.keys())[3]] = (i[3][1])
            predictive_parity_data=classification.copy()

    return predictive_parity_data,thresholds

    # Must complete this function!
    #return predictive_parity_data, thresholds

    #return None, None

    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
   # applied_threshold_value = apply_threshold(categorical_results[j], i)
    #pp_values = get_true_positive_rate(applied_threshold_value)
    #equal[j].append((pp_values, i))
    single_threshold_data = {}
    thresholds = {}
    maximum=0
    applied_threshold_value={}
    import numpy as np
    threshold = np.arange(0.0, 1.0, 0.01)
    for i in threshold:
        for j in categorical_results.keys():
            applied_threshold_value[j] = apply_threshold(categorical_results[j], i)
        #print("applied_threshold_value",applied_threshold_value)
        accuracy = get_total_accuracy(applied_threshold_value)
        if accuracy>maximum:
            maximum=accuracy
            thresh=i
    thresholds=dict.fromkeys(categorical_results,thresh)
    for m in categorical_results.keys():
        single_threshold_data[m]=apply_threshold(categorical_results[m],thresh)
    # Must complete this function!
    return single_threshold_data, thresholds

    #return None, None