
!pip install pm4py 
!pip install graphviz

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from matplotlib import pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
import pydotplus as pydot
from keras.layers import Dropout
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.conversion.log import converter as xes_converter
from pm4py.objects.log.importer.xes import importer as xes_importer

# Set parameter
prefix_size = 11

"""#**Preprocessing training data**"""

log_xes_train = xes_importer.apply('/content/Helpdesk_Train_OOT.xes')
log_train = xes_converter.apply(log_xes_train, variant=xes_converter.Variants.TO_DATA_FRAME)


"""## Data preparation"""

# Remove the word 'Case' in column Case ID
log_train["Case ID"] = log_train["Case ID"].str.replace('Case', '')

# Change columns names
log_train.rename(columns={'Case ID': 'case'}, inplace=True)
log_train.rename(columns={'Complete Timestamp': 'timestamp'}, inplace=True)
log_train.rename(columns={'Activity': 'event'}, inplace=True)

log_train['event'].unique()
# 12 activities in the event log

# Printing 2 column names: activity Y is the next activity following activity X
_ncols_train = ('X_train', 'Y_train') 
# Set active case equal to NULL
_activeCase_train = "NULL"
# Make a dataframe with columns X and Y
maindfObj_train = pd.DataFrame([], columns=_ncols_train)
_tempxy_train = []

def create_input_output_train(xy, case_id):
    global maindfObj_train      # creating a global variable to be used outside of the function as well.
    values = []           # Define Empty List
    xList = [];
    values.append(("NULL", xy[0]))    #adding the value "NULL" to the list
    i = 0
    while i < len(xy):
        try:
            xList = xy[0: i+1]
            xList.insert(0, "NULL")
            values.append((xList, xy[i + 1]))
        except:
            xList = xy[0: i+1]
            xList.insert(0, "NULL")
            values.append((xList, "END"))
        i = i + 1
    subdfObj_train = pd.DataFrame(values, columns=_ncols_train)
    maindfObj_train = maindfObj_train.append(subdfObj_train)

for index, row in log_train.iterrows():      
      if 'case' in row and (row['case'] == _activeCase_train or _activeCase_train == "NULL"):
          concatenatedString = row['event']
          _tempxy_train.append(concatenatedString)
          _activeCase_train = row['case']
      else:
        create_input_output_train(_tempxy_train, _activeCase_train)
        _activeCase_train = row['case']
        _tempxy_train.clear()
        concatenatedString = row['event']
        _tempxy_train.append(concatenatedString)

# Table with columns X and Y
event_log_train = maindfObj_train
event_log_train

event_log_train["Y_train"].unique()
# Including END activities, there are 13 possible next activities

"""## Encoding and padding"""

tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n') #filters out punctuation 
tokenizer.fit_on_texts(event_log_train['X_train'])  #dictionary creation
X_train = tokenizer.texts_to_sequences(event_log_train['X_train'])
word_index = tokenizer.word_index
print(word_index)
print('Found %s unique tokens.' % len(word_index))

X_train = pad_sequences(X_train, maxlen = prefix_size, padding = 'pre', truncating = 'pre', value=0)
print(X_train.shape)
print(X_train)
# original prefix length is 16

# Convert categorical data into dummy or indicator variables.
Y_train = pd.get_dummies(event_log_train['Y_train'])
print(Y_train.shape)
print(Y_train.columns)

# Add missing activities to match Y column shapes
# Add activity column at the correct place 
# Add value 0 for this activity as it never occurs
Y_train.insert(3, 'DUPLICATE', 0)

Y_train.insert(8, 'Require upgrade', 0) 
print(Y_train.columns)
print(Y_train.shape)

"""#**Preprocess validation data**"""

# Import validation set
log_xes_val = xes_importer.apply('/content/Helpdesk_Val_OOT.xes')
log_val = xes_converter.apply(log_xes_val, variant=xes_converter.Variants.TO_DATA_FRAME)

"""## Data preparation"""

# Remove the word 'Case' in column Case ID
log_val["Case ID"] = log_val["Case ID"].str.replace('Case', '')

# Change columns names
log_val.rename(columns={'Case ID': 'case'}, inplace=True)
log_val.rename(columns={'Complete Timestamp': 'timestamp'}, inplace=True)
log_val.rename(columns={'Activity': 'event'}, inplace=True)

log_val["event"].unique()
# Contains all 12 activities

# Printing 2 column names: activity Y is the next activity following activity X
_ncols_val = ('X_val', 'Y_val') 
# Set active case equal to NULL
_activeCase_val = "NULL"
# Make a dataframe with columns X and Y
maindfObj_val = pd.DataFrame([], columns=_ncols_val)
_tempxy_val = []

def create_input_output_val(xy, case_id):
    global maindfObj_val      # creating a global variable to be used outside of the function as well.
    values = []           # Define Empty List
    xList = [];
    values.append(("NULL", xy[0]))    #adding the value "NULL" to the list
    i = 0
    while i < len(xy):
        try:
            xList = xy[0: i+1]
            xList.insert(0, "NULL")
            values.append((xList, xy[i + 1]))
        except:
            xList = xy[0: i+1]
            xList.insert(0, "NULL")
            values.append((xList, "END"))
        i = i + 1
    subdfObj_val = pd.DataFrame(values, columns=_ncols_val)
    maindfObj_val = maindfObj_val.append(subdfObj_val)

# Building traces = kijken naar elke rij en bij elke nieuwe case wordt er een nieuwe trace gestart (=else). 
# Indien hetzelfde case nummer als de vorige, dan wordt dit aan de bestaande trace toegevoegd (=if).
for index, row in log_val.iterrows():      
      if 'case' in row and (row['case'] == _activeCase_val or _activeCase_val == "NULL"):
          concatenatedString = row['event']
          _tempxy_val.append(concatenatedString)
          _activeCase_val = row['case']
      else:
        create_input_output_val(_tempxy_val, _activeCase_val)
        _activeCase_val = row['case']
        _tempxy_val.clear()
        concatenatedString = row['event']
        _tempxy_val.append(concatenatedString)

# Table with columns X and Y
event_log_val = maindfObj_val
event_log_val

event_log_val['Y_val'].unique()
# Contains all activities + END activity = 13 activities

"""## Encoding and padding """

# Convert activities/events to tokens.
X_val = tokenizer.texts_to_sequences(event_log_val['X_val'])

# X_val_original = pad_sequences(X_val)
# print(X_val_original.shape)
# orginal prefix length = 9

# Use padding to ensure all sequences are of same length.
X_val = pad_sequences(X_val, maxlen = prefix_size, padding = 'pre', truncating = 'pre', value = 0)
print(X_val.shape)
print(X_val)

# Convert categorical data into dummy or indicator variables.
Y_val = pd.get_dummies(event_log_val['Y_val'])
print(Y_val.shape)
print(Y_val.columns)

# Add missing activities to match Y column shapes
# Add activity column at the correct place 
# Add value 0 for this activity as it never occurs
Y_val.insert(3, 'DUPLICATE', 0)
Y_val.columns

Y_val.insert(11, 'Schedule intervention', 0)
print(Y_val.columns)
print(Y_val.shape)

"""# **Preprocess test data**"""

log_xes_test = xes_importer.apply('/content/Helpdesk_Test_OOT.xes')
log_test = xes_converter.apply(log_xes_test, variant=xes_converter.Variants.TO_DATA_FRAME)

"""## Data preparation """

# Remove the word 'Case' in column Case ID
log_test["Case ID"] = log_test["Case ID"].str.replace('Case', '')

# Change columns names
log_test.rename(columns={'Case ID': 'case'}, inplace=True)
log_test.rename(columns={'Complete Timestamp': 'timestamp'}, inplace=True)
log_test.rename(columns={'Activity': 'event'}, inplace=True)

log_test["event"].unique()

# Printing 2 column names: activity Y is the next activity following activity X
_ncols_test = ('X_test', 'Y_test') 
# Set active case equal to NULL
_activeCase_test = "NULL"
# Make a dataframe with columns X and Y
maindfObj_test = pd.DataFrame([], columns=_ncols_test)
_tempxy_test = []

def create_input_output_test(xy, case_id):
    global maindfObj_test      # creating a global variable to be used outside of the function as well.
    values = []           # Define Empty List
    xList = [];
    values.append(("NULL", xy[0]))    #adding the value "NULL" to the list
    i = 0
    while i < len(xy):
        try:
            xList = xy[0: i+1]
            xList.insert(0, "NULL")
            values.append((xList, xy[i + 1]))
        except:
            xList = xy[0: i+1]
            xList.insert(0, "NULL")
            values.append((xList, "END"))
        i = i + 1
    subdfObj_test = pd.DataFrame(values, columns=_ncols_test)
    maindfObj_test = maindfObj_test.append(subdfObj_test)


for index, row in log_test.iterrows():      
      if 'case' in row and (row['case'] == _activeCase_test or _activeCase_test == "NULL"):
          concatenatedString = row['event']
          _tempxy_test.append(concatenatedString)
          _activeCase_test = row['case']
      else:
        create_input_output_test(_tempxy_test, _activeCase_test)
        _activeCase_test = row['case']
        _tempxy_test.clear()
        concatenatedString = row['event']
        _tempxy_test.append(concatenatedString)

# Table with columns X and Y
event_log_test = maindfObj_test
event_log_test

X_test = tokenizer.texts_to_sequences(event_log_test['X_test'])
X_test = pad_sequences(X_test, maxlen = prefix_size, padding = 'pre', truncating = 'pre', value = 0)
print(X_test.shape)
print(X_test)

Y_test = pd.get_dummies(event_log_test['Y_test'])
print(Y_test.shape)
print(Y_test.columns)

# Add missing activities to match Y column shapes
# Add activity column at the correct place 
# Add value 0 for this activity as it never occurs
Y_test.insert(5, 'INVALID', 0)

Y_test.insert(7, 'RESOLVED', 0)
print(Y_test.columns)
print(Y_test.shape)

Y_test

"""# **Building Model Architecture**"""

# DIFFERS FROM DATASET
# Required arguments for the embedding layer
MAX_NB_WORDS = 15 # vocab_size is equal to the number of unique tokens
EMBEDDING_DIM = 6 # paper uses 50 depending on used data, opmerking Jari: raar dat embedding size groter dan vocab size

# Required to downgrade numpy to run: pip install numpy==1.19.5
model = Sequential()
embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]) 
model.add(embedding_layer)
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(Y_train.shape[1], activation='softmax'))

"""# **Compiling Model**"""

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
plot_model(model, to_file='model.png', show_shapes=True)

"""# **Training LSTM Model**"""

history = model.fit(X_train, Y_train, validation_data=(X_val,Y_val),  epochs=10, batch_size=64, verbose=2)

#loss, accuracy= model.evaluate(X, Y)
loss, accuracy= model.evaluate( X_val, Y_val)
print('Loss: {:0.3f}\n  Accuracy: {:0.3f}\n '.format(loss, (accuracy*100)))

#print(embedding_layer.get_weights()[0].shape)
from matplotlib import pyplot as plt
plt.style.use('ggplot')

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.show()

# save model to single file
model.save('LSTM_Helpdesk.h5')

