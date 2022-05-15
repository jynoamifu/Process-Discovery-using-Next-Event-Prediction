# **Matrix and Visualization**

## **Directly-follows matrix**

# load prediction from single file
from tensorflow import keras
predictions = pd.read_csv('Predictions_HelpDesk.csv', na_filter=True)
predictions

headers = list(predictions.columns.values)
predictions = predictions.drop([headers[0]], axis=1) #drop first column
predictions.head()

# Find the last activity for each trace
predictions['last_activity'] = predictions.iloc[:, :prefix_size].ffill(axis=1).iloc[:, -1].fillna('NULL')
print (predictions['last_activity'])

#Take the mean for every activity
df = predictions[['last_activity','Assign seriousness', 'Closed', 'Create SW anomaly', 'DUPLICATE', 'END',
       'INVALID', 'Insert ticket', 'RESOLVED', 'Require upgrade',
       'Resolve SW anomaly', 'Resolve ticket', 'Schedule intervention',
       'Take in charge ticket', 'VERIFIED', 'Wait']]
prob = df.groupby(df['last_activity']).mean().reset_index()

#Check whether the probabilities sum up to 100%
check = prob.sum(axis=1)
check

prob = prob.set_index("last_activity")
prob = prob.round(2)
prob

"""### **DFG**"""

import graphviz

"""#### All above threshold"""

dot_all = graphviz.Digraph('state transition diagram (all edges)')

for a in prob.index:
  dot_all.node(a) # Add the activities as nodes

dot_all

THRESHOLD = 1 # %

for start_node, row in prob.iterrows(): # Loop over the dataframe rows
    for end_node, probability in row.items():  
      if probability > THRESHOLD:
        dot_all.edge(start_node, end_node, str(probability) + "%") # Add an edge to the graph

dot_all

filename=dot_all.render(filename='DFG_helpdesk_1%')
