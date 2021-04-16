from math import inf, log, exp, sqrt
from functools import reduce
from random import random, randrange, seed, choice, uniform
import logging
import os.path
import time

def gsemo_algorithm(model):
    """The GSEMO algorithm for maximizing an (approximately) submodular
    utility function.

    Args:
        model (models.Model): The submodular model to use

    Returns:
        pair (best_res,best_value) of type (list of int/None, float).
        The first component is the matching, the second its queried value in
        the model.
    """
    # in element we use matrix to show all possible agent-locality pairs
    class ArchivedElem(object):
        def __init__(self, f1_value, f2_value, element, locality_per_agent):
            super(ArchivedElem, self).__init__()
            self.f1_value = f1_value
            self.f2_value = f2_value
            self.element = element
            self.locality_per_agent = locality_per_agent

    p = 1.0 / (model.num_agents * len(model.locality_caps))
    init_elem = [[0 for _ in range(len(model.locality_caps))] for _ in range(model.num_agents)]

    for i in range(len(init_elem)):
        for j in range(len(init_elem[i])):
            if uniform(0,1) < p:
                init_elem[i][j] = 1 - init_elem[i][j]

    f1_init = f2_init = 0
    init_caps = [0 for _ in range(len(model.locality_caps))]
    init_locality_per_agent = [None for _ in range(model.num_agents)]
    for i in range(len(init_elem)):
        cnt = 0
        for j in range(len(init_elem[i])):
            if init_elem[i][j] == 1:
                init_locality_per_agent[i] = j
                cnt += 1
                init_caps[j] += 1
            else:
                f2_init += 1
        if cnt > 1:
            f1_init = -1
    for i in range(len(init_caps)):
        if init_caps[i] > model.locality_caps[i]:
            f1_init = -1
    if f1_init != -1:
        f1_init = model.utility_for_matching(init_locality_per_agent)
    archived_set = [ArchivedElem(f1_init,f2_init,init_elem,init_locality_per_agent)]

    # # logging
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    # log_path = os.path.dirname(os.getcwd()) + '/Logs1/'
    # log_name = log_path + rq + '.log'
    # logfile = log_name
    # fh = logging.FileHandler(logfile, mode='w')
    # fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

    T = model.num_agents * len(model.locality_caps) * 10000
    for it in range(T):
        selected = choice(archived_set)
        selected_elem = [[0 for _ in range(len(model.locality_caps))] for _ in range(model.num_agents)]
        for i in range(len(selected_elem)):
            for j in range(len(selected_elem[i])):
                selected_elem[i][j] = selected.element[i][j]

        for i in range(len(selected_elem)):
            for j in range(len(selected_elem[i])):
                if uniform(0,1) < p:
                    selected_elem[i][j] = 1 - selected_elem[i][j]

        f1_selected = f2_selected = 0
        selected_caps = [0 for _ in range(len(model.locality_caps))]
        locality_per_agent = [None for _ in range(model.num_agents)]
        for i in range(len(selected_elem)):
            f = 0
            for j in range(len(selected_elem[i])):
                if selected_elem[i][j] == 1:
                    locality_per_agent[i] = j
                    f += 1
                    selected_caps[j] += 1
                else:
                    f2_selected += 1
            if f > 1:
                f1_selected = -1
        for i in range(len(selected_caps)):
            if selected_caps[i] > model.locality_caps[i]:
                f1_selected = -1
        if f1_selected != -1:
            f1_selected = model.utility_for_matching(locality_per_agent)

        flag = True
        for e in archived_set:
            if ((e.f1_value > f1_selected) and (e.f2_value >= f2_selected)) or ((e.f1_value >= f1_selected) and (e.f2_value > f2_selected)):
                flag = False
                break

        if flag == True:
            archived_set = [e for e in archived_set if not ((f1_selected >= e.f1_value) and (f2_selected >= e.f2_value))]
            # for e in archived_set:
            #     if f1_selected >= e.f1_value and f2_selected >= e.f2_value:
            #         archived_set.remove(e)
            archived_set.append(ArchivedElem(f1_selected, f2_selected, selected_elem, locality_per_agent))

        # if it % 1e6 == 0:
        #     lena = len(archived_set)
        #     fvalue = -1
        #     loc = [None for _ in range(model.num_agents)]
        #     lenlist = []
        #     f1list = []
        #     f2list = []
        #     for elem in archived_set:
        #         sum0 = 0
        #         for i1 in range(len(elem.element)):
        #             for j1 in range(len(elem.element[i1])):
        #                 if elem.element[i1][j1] == 0:
        #                     sum0 += 1
        #         lenlist.append(sum0)
        #         f1list.append(elem.f1_value)
        #         f2list.append(elem.f2_value)
        #         if fvalue < elem.f1_value:
        #             fvalue = elem.f1_value
        #             loc = elem.locality_per_agent
        #     # logger.info(f'f1list = {f1list}, f2list = {f2list}, lenlist = {lenlist}, max f1 value = {fvalue}, max locality_per_agent = {loc}, locality_caps = {model.locality_caps}, archived_set_size = {lena}, selected_elem = {selected_elem}')

    best_value = -1
    best_res = [None for _ in range(model.num_agents)]
    for e in archived_set:
        if e.f1_value > best_value:
            best_value, best_res = e.f1_value, e.locality_per_agent

    return best_res, model.utility_for_matching(best_res,False)


def greedy_algorithm(model):
    """The greedy algorithm for maximizing an (approximately) submodular
    utility function.

    Args:
        model (models.Model): The submodular model to use

    Returns:
        pair (locality_per_agent,best_value) of type (list of int/None, float).
        The first component is the matching, the second its queried value in
        the model.
    """
    locality_per_agent = [None for _ in range(model.num_agents)]
    caps_remaining = [cap for cap in model.locality_caps]

    for _ in range(min(model.num_agents, sum(caps_remaining))):
        best_pair = None
        best_value = -inf
        for i, match in enumerate(locality_per_agent):
            if match != None:
                continue

            for l, spaces in enumerate(caps_remaining):
                if spaces <= 0:
                    continue

                locality_per_agent[i] = l
                utility = model.utility_for_matching(locality_per_agent)
                locality_per_agent[i] = None

                if utility > best_value:
                    best_pair = (i, l)
                    best_value = utility

        assert best_pair != None
        i, l = best_pair
        locality_per_agent[i] = l
        caps_remaining[l] -= 1

    return locality_per_agent, model.utility_for_matching(locality_per_agent,
                                                          False)
