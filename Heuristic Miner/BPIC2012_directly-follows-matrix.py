
!pip install pm4py
!pip install simpy

from copy import copy
from collections import Counter
from pm4py.util.constants import DEFAULT_VARIANT_SEP
import numpy as np
import pkgutil
import logging
import pm4py
import pandas as pd

from pm4py.objects.log.importer.xes import importer as xes_importer
log = xes_importer.apply('/content/BPI_Challenge_2012.xes')
log

# Printing the start activity in our log
from pm4py.algo.filtering.log.start_activities import start_activities_filter
log_start = start_activities_filter.get_start_activities(log)
log_start

# Printing end activities
from pm4py.algo.filtering.log.end_activities import end_activities_filter
end_activities = end_activities_filter.get_end_activities(log)
end_activities

# Total occurences NULL = 13087

"""# Approach """

from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
heu_net = heuristics_miner.apply_heu(log, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99})

#Visualizing a Heuristic Net (same output as Disco)
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
gviz_heuristic = hn_visualizer.apply(heu_net)
hn_visualizer.view(gviz_heuristic)

"""Transition Matrix"""

freq_dict = heu_net.performance_matrix
freq_dict
# PROBLEM: No start or end activity included here!

freq_dict = pd.DataFrame.from_dict(freq_dict, orient = "index")
freq_dict = freq_dict.fillna(0)
freq_dict

# Get the order of the column names
for col in freq_dict.columns:
    print(col)

# Get the order of the column names
for col in freq_dict.columns:
    print(col)

freq_dict.reset_index(inplace = True)
freq_dict

# Add starting transitions manually
#'A_SUBMITTED': 13087
freq_dict = freq_dict.append({'index':'NULL',	'A_SUBMITTED':13087, 'A_PARTLYSUBMITTED': 0.0, 'A_PREACCEPTED':0.0,
                              'A_DECLINED':0.0, 'W_Afhandelen leads':0.0, 'W_Beoordelen fraude':0.0, 'W_Completeren aanvraag':0.0,
                              'A_ACCEPTED':0.0, 'W_Nabellen offertes':0.0, 'A_CANCELLED':0.0, 'O_SELECTED':0.0, 'A_FINALIZED':0.0,
                              'O_CREATED':0.0, 'O_CANCELLED':0.0, 'O_SENT':0.0, 'W_Nabellen incomplete dossiers':0.0, 'O_SENT_BACK':0.0,
                              'W_Valideren aanvraag':0.0, 'W_Wijzigen contractgegevens':0.0, 'O_DECLINED':0.0, 'A_REGISTERED':0.0,
                              'O_ACCEPTED':0.0, 'A_APPROVED':0.0, 'A_ACTIVATED':0.0}, ignore_index=True)
freq_dict

# Add ending transitions manually
#  'A_CANCELLED': 655,
#  'A_DECLINED': 3429,
#  'A_REGISTERED': 1,
#  'O_CANCELLED': 279,
#  'W_Afhandelen leads': 2234,
#  'W_Beoordelen fraude': 57,
#  'W_Completeren aanvraag': 1939,
#  'W_Nabellen incomplete dossiers': 452,
#  'W_Nabellen offertes': 1290,
#  'W_Valideren aanvraag': 2747,
#  'W_Wijzigen contractgegevens': 4
ending_transitions = [0.0, 0.0, 2234, 1939, 2747, 0.0, 57, 3429, 0.0, 1290, 655, 0.0, 279, 4, 452, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0]
freq_dict["END"] = ending_transitions 
freq_dict

"""Occurences Activity"""

occurences = heu_net.activities_occurrences
occurences

# Transform the count of activities into a dataframe
occurences = pd.DataFrame.from_dict(occurences, orient = "index")
occurences.reset_index(inplace = True)   # Give dataframe an index column
occurences.rename(columns = {0 : 'count'}, inplace = True) # check column name because it was unnamed
occurences

# Add counts of NULL 13087
occurences = occurences.append({'index':'NULL',	'count': 13087}, ignore_index=True)
occurences

"""Probability Matrix"""

probability_heu = pd.merge(occurences, freq_dict,left_on= "index" ,right_on= "index")
probability_heu = probability_heu.set_index("index")
probability_heu

# Divide all transitions by the number of occurences of the source node
# Divide each column by the count column based on index name 
probability_heu = probability_heu.divide(probability_heu["count"], axis = "index")
probability_heu

probability_heu = (probability_heu * 100).round(2)
del probability_heu["count"]
probability_heu

probability_heu.to_csv("prob_from_heu_BPIC2012.csv", index = True)
