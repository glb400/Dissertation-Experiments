from math import sqrt
from random import random, randrange, seed
import matplotlib
import seaborn
import pandas
from models import *
from methods import *
seaborn.set(style="darkgrid")
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman"]
seed(0)

num_agents = 100
num_localities = 10
random_samples = 1000
locality_caps = [10] * num_localities

def _distribute_professions_caps_and_jobs(num_professions):
    assert num_professions <= num_agents
    professions = list(range(num_professions))
    profession_counts = [1 for _ in range(num_professions)]
    for _ in range(num_agents - num_professions):
        prof = randrange(num_professions)
        professions.append(prof)
        profession_counts[prof] += 1
    
    job_numbers = []
    profession_remaining = profession_counts[:]
    jobs_remaining = num_agents
    for cap in locality_caps:
        ps = [0 for _ in range(num_professions)]
        for _ in range(cap):
            a = random()
            for prof in range(num_professions):
                if a < (profession_remaining[prof] / jobs_remaining):
                    ps[prof] += 1
                    profession_remaining[prof] -= 1
                    break
                a -= profession_remaining[prof] / jobs_remaining
            jobs_remaining -= 1
        job_numbers.append(tuple(ps))
    
    assert sum(profession_remaining) == 0
        
    return profession_counts, professions, job_numbers
    
def test_correction(num_professions):
    _, professions, job_numbers = \
        _distribute_professions_caps_and_jobs(num_professions)
    qualification_probabilities = \
        [[random()] * num_localities for _ in range(num_agents)]
    correction_functions = []
    for ps in job_numbers:
        # The default parameters in the lambdas are never used, but are
        # a way of getting Python's peculiar binding behavior to work.
        # See https://docs.python.org/3/faq/programming.html#why-do-
        # lambdas-defined-in-a-loop-with-different-values-all-return-
        # the-same-result for more information.
        correction_functions.append(
            [(lambda x, P=p: min(x, P)) for p in ps])
    model = RetroactiveCorrectionModel(num_agents, locality_caps,
                                       num_professions, professions,
                                       qualification_probabilities,
                                       correction_functions,
                                       random_samples)
    return model

def test_interview(num_professions):
    _, professions, job_numbers = \
        _distribute_professions_caps_and_jobs(num_professions)
    compatibility_probabilities = [random() for _ in range(num_agents)]
    model = InterviewModel(num_agents, locality_caps, num_professions,
                           professions, job_numbers,
                           compatibility_probabilities, random_samples)
    return model

def test_coordination(num_professions):
    profession_counts, professions, job_numbers = \
        _distribute_professions_caps_and_jobs(num_professions)
    locality_num_jobs = locality_caps
    compatibility_probabilities = [] 
    for i, prof in enumerate(professions):
        competency = random()
        probs = []
        for ps in job_numbers:
            a = []
            for prof2, prof2nums in enumerate(ps):
                if prof2 == prof:
                    a += [competency] * prof2nums
                else:
                    a += [0] * prof2nums
            probs.append(a)
        compatibility_probabilities.append(probs)
    model = CoordinationModel(num_agents, locality_caps,
                              locality_num_jobs,
                              compatibility_probabilities,
                              random_samples)
    return model

settings = {"correction": test_correction, "interview": test_interview,
            "coordination": test_coordination}

data = []

def sample(logger, setting, num_professions):
    m = settings[setting](num_professions)
    greedy = greedy_algorithm(m)[1]
    gsemo = gsemo_algorithm(m)[1]
    datum = {}
    datum["number of professions"] = num_professions
    datum["greedy"] = greedy
    datum["gsemo"] = gsemo
    if greedy > 0.0005:
        datum["gsemo / greedy"] = gsemo / greedy
    else:
        datum["gsemo / greedy"] = None
    datum["model"] = setting
    print(f'gsemo = {gsemo}',f' greedy = {greedy}', f' gsemo / greedy = {gsemo} / {greedy}')
    logger.info(f'gsemo = {gsemo}, greedy = {greedy}, gsemo / greedy = {gsemo} / {greedy}')
    data.append(datum)
    return datum

from datetime import datetime
import logging
import os.path
import time
logger = logging.getLogger()
logger.setLevel(logging.INFO)
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = os.path.dirname(os.getcwd()) + '/Logs6/'
log_name = log_path + rq + '.log'
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

for _ in range(2):
    for num_professions in [2, 3, 5, 8, 10, 15]:
        for setting in settings:
            sample(logger, setting, num_professions)
            print(datetime.now(), setting, num_professions, len(data))
from pickle import dumps
dumps(data)

def _format_y(ratio):
    return f"{ratio-1:,.1%}"

def plot():
    d = pandas.DataFrame(data)
    g = seaborn.catplot(x="number of professions", y="gsemo / greedy",
                        hue="model",
                        hue_order=["correction", "interview",
                                   "coordination"],
                        data=d)
    for ax in g.axes[0]:
        vals = ax.get_yticks()
        ax.set_yticklabels([_format_y(x) for x in vals])
        ax.set_ylabel("improvement of gsemo over greedy")
    g.savefig("num_professions.pdf")

plot()