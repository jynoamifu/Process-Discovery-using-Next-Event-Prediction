
!pip install pm4py

import pm4py

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.filtering.log.timestamp import timestamp_filter

import numpy as np

import math

import pandas as pd
import numpy as np

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
    test_log  = timestamp_filter.filter_traces_contained(log, split_point, max_time)
  else:
    train_log = timestamp_filter.filter_traces_intersecting(log, min_time, split_point)
    test_log  = timestamp_filter.filter_traces_contained(log, split_point, max_time)
  print(min_time)
  print(split_point)
  print(max_time)
  print("size log:",len(log), "Deleted due to overlap:", (len(log) - len(train_log) - len(test_log)))
  return(train_log, test_log)


def import_and_save_OOT_split(log_location, fraction, remove_overlap, namelog):
  tr, te = OOT_split(log_location, fraction, remove_overlap)
  trainname = namelog + "Train_OOT"+".xes"
  testname = namelog + "Test_OOT"+".xes"
  xes_exporter.apply(tr, trainname)
  xes_exporter.apply(te, testname)


"""# **Example using Helpdesk data**"""

from google.colab import files
upload = files.upload()

log_csv = pd.read_csv("HelpDesk.csv")
log_csv

log_csv = pm4py.format_dataframe(log_csv, case_id='Case ID', activity_key='Activity', timestamp_key='Complete Timestamp')
start_activities = pm4py.get_start_activities(log_csv)
end_activities = pm4py.get_end_activities(log_csv)
print("Start activities: {}\nEnd activities: {}".format(start_activities, end_activities))

from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

log = log_converter.apply(log_csv)
log

from pm4py.objects.log.exporter.xes import exporter as xes_exporter
xes_exporter.apply(log, 'Helpdesk.xes')

OOT_split(log_location= '/content/Helpdesk.xes', fraction=0.8, remove_overlap=True)

import_and_save_OOT_split(log_location='/content/Helpdesk.xes', fraction=0.8, remove_overlap=True, namelog='Helpdesk_2012')
