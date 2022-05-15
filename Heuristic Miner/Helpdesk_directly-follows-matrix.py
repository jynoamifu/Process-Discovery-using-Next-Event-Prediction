# Install required packages
!pip install pm4py 
!pip install graphviz

from copy import copy
from collections import Counter
from pm4py.util.constants import DEFAULT_VARIANT_SEP
import numpy as np
import pkgutil
import logging
import pm4py
import pandas as pd

# Load event log
from pm4py.objects.log.importer.xes import importer as xes_importer
log = xes_importer.apply('/content/Helpdesk.xes')
log

# Printing the start activity in our log
from pm4py.algo.filtering.log.start_activities import start_activities_filter
log_start = start_activities_filter.get_start_activities(log)
log_start  

# Printing end activities
from pm4py.algo.filtering.log.end_activities import end_activities_filter
end_activities = end_activities_filter.get_end_activities(log)
end_activities

# Total occurences NULL = 4580

from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
heu_net = heuristics_miner.apply_heu(log, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99})

# Visualizing a Heuristic Net (same output as Disco)
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
gviz_heuristic = hn_visualizer.apply(heu_net)
hn_visualizer.view(gviz_heuristic)

## Create transition matrix ##
freq_dict = heu_net.performance_matrix
freq_dict
# PROBLEM: No start or end activity included here!

freq_dict = pd.DataFrame.from_dict(freq_dict, orient = "index")
freq_dict = freq_dict.fillna(0)

# Add column for Insert Ticket
freq_dict.insert(6, "Insert ticket", 0.0, allow_duplicates = False)
freq_dict.reset_index(inplace = True)

# Add starting transitions manually
#'Assign seriousness': 4384,
#'Create SW anomaly': 1,
#'Insert ticket': 118,
#'Resolve ticket': 2,
#'Take in charge ticket': 74,
#'Wait': 1
# index	Take in charge ticket	Assign seriousness	Resolve ticket	Wait	Create SW anomaly	Require upgrade	Insert ticket	Schedule intervention	Resolve SW anomaly	Closed	RESOLVED	VERIFIED	INVALID	DUPLICATE
freq_dict = freq_dict.append({'index':'NULL',	'Take in charge ticket':74,	'Assign seriousness':4384,	'Resolve ticket':2 ,	'Wait':1 ,	'Create SW anomaly':1 ,	'Require upgrade':0	,'Insert ticket':118, 
                              'Schedule intervention':0	,'Resolve SW anomaly':0	,'Closed':0	,'RESOLVED':0	,'VERIFIED':0	,'INVALID':0	,'DUPLICATE':0}, ignore_index=True)


# Add ending transitions manually
# 'Closed': 4557,
# 'Require upgrade': 3,
# 'Resolve ticket': 10,
# 'Take in charge ticket': 1,
# 'VERIFIED': 1,
# 'Wait': 8
ending_transitions = [0.0, 3, 8, 0.0, 0.0, 0.0, 0.0, 10, 0.0, 0.0, 0.0, 1, 4557, 0.0, 0.0]
freq_dict["END"] = ending_transitions 

## Activity Occurences ## 
occurences = heu_net.activities_occurrences

# Transform the count of activities into a dataframe
occurences = pd.DataFrame.from_dict(occurences, orient = "index")
occurences.reset_index(inplace = True)   # Give dataframe an index column
occurences.rename(columns = {0 : 'count'}, inplace = True) # check column name because it was unnamed

# Add counts of NULL 4580
occurences = occurences.append({'index':'NULL',	'count': 4580}, ignore_index=True)

## Directly-follows matrix ## 
probability_heu = pd.merge(occurences, freq_dict,left_on= "index" ,right_on= "index")
probability_heu = probability_heu.set_index("index")

# Divide all transitions by the number of occurences of the source node
# Divide each column by the count column based on index name 
probability_heu = probability_heu.divide(probability_heu["count"], axis = "index")

probability_heu = (probability_heu * 100).round(2)
del probability_heu["count"]
probability_heu

probability_heu.to_csv("prob_from_heu_helpdesk.csv", index = True)
