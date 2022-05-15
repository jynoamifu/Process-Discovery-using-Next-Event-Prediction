
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

#Noami and Julia: I recommend you use remove_overlap = True
#Chen, Charilaos, Tove I recommend you use remove_overlap = False
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


def random_split(log_location, fraction): 
  #returns train and test log where the traces are split randomly
  #fraction = fraction of traces in the training log
  variant = xes_importer.Variants.ITERPARSE
  parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
  log = xes_importer.apply(log_location, variant=variant, parameters=parameters)
  place = int(len(log)*fraction)
  print(place)
  indices = np.random.permutation(len(log))
  train_idx, test_idx = indices[:place], indices[place:]
  train_log = [log[i] for i in train_idx]
  test_log = [log[i] for i in test_idx]
  train_log = EventLog(train_log)
  test_log = EventLog(test_log)
  return train_log, test_log

def k_fold_split_random(log_location, k): 
  #returns two lists of length k: the training and test logs
  #we split the log randomly into k folds
  #the test log is each time one of these folds
  #the corresponding training log is then the rest of the log
  variant = xes_importer.Variants.ITERPARSE
  parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
  log = xes_importer.apply(log_location, variant=variant, parameters=parameters)
  print(len(log))
  #define sizes of folds
  foldsize = math.floor(len(log)/k)
  places = []
  places.append(0)
  for i in range(0, k):
    place = (i+1)*foldsize
    places.append(place)
  #fix these last traces not included   
  left = int(len(log) - foldsize*k)
  for l in range(1, left):
    places[l] = places[l] + l
  for k in range(left, len(places)):
    places[k] = places[k] + left
  print(places)    
  indices = np.random.permutation(len(log))
  train_logs = []
  test_logs = []
  for j in range(0, k):
    tr = [log[i] for i in train_idx]
    te = [log[i] for i in test_idx]
    train_log = EventLog(tr)
    test_log = EventLog(te)
    train_logs.append(train_log)
    test_logs.append(test_log)
  return(train_logs, test_logs)

def import_and_save_OOT_split(log_location, fraction, remove_overlap, namelog):
  tr, te = OOT_split(log_location, fraction, remove_overlap)
  trainname = namelog + "Train_OOT"+".xes"
  testname = namelog + "Test_OOT"+".xes"
  xes_exporter.apply(tr, trainname)
  xes_exporter.apply(te, testname)

def import_and_save_random_split(log_location, fraction, remove_overlap, namelog):
  tr, te = random_split(log_location, fraction)
  trainname = namelog + "Train_Random"+".xes"
  testname = namelog + "Test_Random"+".xes"
  xes_exporter.apply(tr, trainname)
  xes_exporter.apply(te, testname)

def import_and_save_k_fold(log_location, k, namelog):
  tr, te = k_fold_split_random(log_location, k)
  for i in range(0, k):
    trainname = namelog + "Train_"+ str(k)+ "fold_" + str(i+1)+".xes"
    testname = namelog + "Test_"+ str(k)+ "fold_" + str(i+1)+".xes"
    xes_exporter.apply(tr[i], trainname)
    xes_exporter.apply(te[i], testname)

"""# ** Helpdesk data**"""

from google.colab import files
upload = files.upload()

log_csv = pd.read_csv("HelpDesk_Online.csv")
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
