
!pip install pm4py

import pm4py

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.filtering.log.timestamp import timestamp_filter

import numpy as np

import math

"""**Functions** """

def get_timestamp_for_OOT_split(log, fraction): 
  #returns the timestamp where the train-test split is done
  #fraction = fraction of traces in the training log
  place = int(len(log)*fraction)
  return(log[place][0]['time:timestamp'])

def OOT_split(log_location, fraction, remove_overlap): 
  #returns train and test log for an out-of-time split
  #fraction = fraction of traces in the training log
  variant = xes_importer.Variants.ITERPARSE
  parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
  log = xes_importer.apply(log_location, variant=variant, parameters=parameters)
  min_time = min(e['time:timestamp'] for t in log for e in t).replace(tzinfo=None)
  max_time = max(e['time:timestamp'] for t in log for e in t).replace(tzinfo=None)
  split_point = get_timestamp_for_OOT_split(log, fraction).replace(tzinfo=None)
  if remove_overlap == True:
    train_log = timestamp_filter.filter_traces_contained(log, min_time, split_point)
    val_log  = timestamp_filter.filter_traces_contained(log, split_point, max_time)
  else:
    train_log = timestamp_filter.filter_traces_intersecting(log, min_time, split_point)
    val_log  = timestamp_filter.filter_traces_contained(log, split_point, max_time)
  print(min_time)
  print(split_point)
  print(max_time)
  print("size log:",len(log), "Deleted due to overlap:", (len(log) - len(train_log) - len(val_log)))
  return(train_log, val_log)


def import_and_save_OOT_split(log_location, fraction, remove_overlap, namelog):
  tr, te = OOT_split(log_location, fraction, remove_overlap)
  trainname = namelog + "Train_OOT"+".xes"
  valname = namelog + "Val_OOT"+".xes"
  xes_exporter.apply(tr, trainname)
  xes_exporter.apply(te, valname)


"""# **Example: Helpdesk Data**"""

OOT_split(log_location='/content/Helpdesk_2012Train_OOT.xes', fraction=0.8, remove_overlap=True)

import_and_save_OOT_split(log_location='/content/Helpdesk_2012Train_OOT.xes', fraction=0.8, remove_overlap=True, namelog='Helpdesk_')
