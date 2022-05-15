
!pip install pm4py

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

"""# **parameter**"""

prefix_size = 16

"""#**Preprocess training data**

## Import data
"""

from pm4py.objects.conversion.log import converter as xes_converter
from pm4py.objects.log.importer.xes import importer as xes_importer

log_xes_train = xes_importer.apply('/content/BPI_Challenge_2012Train_OOT.xes')
log_train = xes_converter.apply(log_xes_train, variant=xes_converter.Variants.TO_DATA_FRAME)

log_train

"""## Data preparation"""

# DIFFERS FROM DATASET
# Change columns names
log_train.rename(columns={'time:timestamp': 'timestamp'}, inplace=True)
log_train.rename(columns={'case:concept:name': 'case'}, inplace=True)
log_train.rename(columns={'concept:name': 'event'}, inplace=True)


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

# Building traces = kijken naar elke rij en bij elke nieuwe case wordt er een nieuwe trace gestart (=else). 
# Indien hetzelfde case nummer als de vorige, dan wordt dit aan de bestaande trace toegevoegd (=if).
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

# Check the number of activities ! this is not the order of the dummificated column
event_log_train['Y_train'].unique()
#25 unique values, bevat alle activiteiten + END activity

"""## Encoding and padding"""

tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n') #filters out punctuation 
tokenizer.fit_on_texts(event_log_train['X_train'])  #dictionary creation
#Transform each text in texts to a sequence of integers
X_train = tokenizer.texts_to_sequences(event_log_train['X_train'])
word_index = tokenizer.word_index
print(word_index)
print('Found %s unique tokens.' % len(word_index))

# Pad
X_train = pad_sequences(X_train, maxlen = prefix_size, padding = 'pre', truncating = 'pre', value=0)
print(X_train.shape)
print(X_train)

# Convert categorical data into dummy or indicator variables.
Y_train = pd.get_dummies(event_log_train['Y_train'])
print(Y_train.shape)
print(Y_train.columns)

X_train

Y_train

"""#**Preprocess validation data**

## Import data
"""


from pm4py.objects.conversion.log import converter as xes_converter
from pm4py.objects.log.importer.xes import importer as xes_importer

log_xes_val = xes_importer.apply('/content/BPI_Challenge_2012Val_OOT.xes')
log_val = xes_converter.apply(log_xes_val, variant=xes_converter.Variants.TO_DATA_FRAME)

log_val

"""## Data preparation"""

# DIFFERS FROM DATASET
# Change columns names
log_val.rename(columns={'time:timestamp': 'timestamp'}, inplace=True)
log_val.rename(columns={'case:concept:name': 'case'}, inplace=True)
log_val.rename(columns={'concept:name': 'event'}, inplace=True)

log_val

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
# missing one activity "w_wijzigen contractgegevens"

"""## Encoding and padding"""

# Convert activities/events to tokens.
X_val = tokenizer.texts_to_sequences(event_log_val['X_val'])

# Use padding to ensure all sequences are of same length.
X_val = pad_sequences(X_val, maxlen = prefix_size, padding = 'pre', truncating = 'pre', value = 0)
print(X_val.shape)
print(X_val)

# Convert categorical data into dummy or indicator variables.
Y_val = pd.get_dummies(event_log_val['Y_val'])
print(Y_val.shape)
print(Y_val.columns)

# Add missing activity 
Y_val.insert(24, 'W_Wijzigen contractgegevens', 0)
print(Y_val.shape)
print(Y_val.columns)

"""# **Preprocess test data**

## Import data
"""

from pm4py.objects.conversion.log import converter as xes_converter
from pm4py.objects.log.importer.xes import importer as xes_importer

log_xes_test = xes_importer.apply('/content/BPI_Challenge_2012Test_OOT.xes')
log_test = xes_converter.apply(log_xes_test, variant=xes_converter.Variants.TO_DATA_FRAME)

log_test

"""## Data preparation"""

# DIFFERS FROM DATASET
# Change columns names
log_test.rename(columns={'time:timestamp': 'timestamp'}, inplace=True)
log_test.rename(columns={'case:concept:name': 'case'}, inplace=True)
log_test.rename(columns={'concept:name': 'event'}, inplace=True)

log_test

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

event_log_test['Y_test'].unique()
# Missing one activity

"""## Encoding and padding"""

# Convert activities/events to tokens.
X_test = tokenizer.texts_to_sequences(event_log_test['X_test'])

# Use padding to ensure all sequences are of same length.
X_test = pad_sequences(X_test, maxlen = prefix_size, padding = 'pre', truncating = 'pre', value = 0)
print(X_test.shape)
print(X_test)

# Convert categorical data into dummy or indicator variables.
Y_test = pd.get_dummies(event_log_test['Y_test'])
print(Y_test.shape)
print(Y_test.columns)

# Add missing activity at the right place
Y_test.insert(24, 'W_Wijzigen contractgegevens', 0)
print(Y_test.shape)
print(Y_test.columns)

"""# **Building Model Architecture**"""

# DIFFERS FROM DATASET
# Required arguments for the embedding layer
MAX_NB_WORDS = 26 # vocab_size is equal to the number of unique tokens
EMBEDDING_DIM = 12 # paper uses 50 depending on used data, opmerking Jari: raar dat embedding size groter dan vocab size

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

# verbose = 2 shows the epoch training progress 
history = model.fit(X_train, Y_train, validation_data=(X_val,Y_val),  epochs=11, batch_size=64, verbose=2)

# save model to single file
model.save('LSTM_BPIC2012_prefixlen16.h5')

"""#**Model Evaluation**"""

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

"""#**Next Activity Predictions** """

#Make prediction on X_test to predict next activity 
yhat = model.predict(X_test, verbose=2)
yhat

colName = ['A_ACCEPTED', 'A_ACTIVATED', 'A_APPROVED', 'A_CANCELLED', 'A_DECLINED',
       'A_FINALIZED', 'A_PARTLYSUBMITTED', 'A_PREACCEPTED', 'A_REGISTERED',
       'A_SUBMITTED', 'END', 'O_ACCEPTED', 'O_CANCELLED', 'O_CREATED',
       'O_DECLINED', 'O_SELECTED', 'O_SENT', 'O_SENT_BACK',
       'W_Afhandelen leads', 'W_Beoordelen fraude', 'W_Completeren aanvraag',
       'W_Nabellen incomplete dossiers', 'W_Nabellen offertes',
       'W_Valideren aanvraag', 'W_Wijzigen contractgegevens']

dfObj = pd.DataFrame(yhat*100, columns = colName)
Seq_Series=event_log_test.X_test.apply(pd.Series) # Constructs series of the X column of the XY table
dfObj.reset_index(drop=True, inplace=True)
Seq_Series.reset_index(drop=True, inplace=True)
predictions = pd.concat([Seq_Series, dfObj], axis=1)
predictions

predictions.to_csv(path_or_buf = "Predictions_BPIC2012.csv", index = True)
