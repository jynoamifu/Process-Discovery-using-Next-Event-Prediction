
#**Matrix & Visualization**

## Directly-follows matrix

# load prediction from single file
from tensorflow import keras
predictions = pd.read_csv('Predictions_BPIC2012.csv', na_filter=True)
predictions

# Find the last activity for each trace
predictions['last_activity'] = predictions.iloc[:, :prefix_size].ffill(axis=1).iloc[:, -1].fillna('NULL')
print (predictions['last_activity'])

#Take the mean for every activity
df = predictions[['last_activity','A_ACCEPTED', 'A_ACTIVATED', 'A_APPROVED', 'A_CANCELLED', 'A_DECLINED',
       'A_FINALIZED', 'A_PARTLYSUBMITTED', 'A_PREACCEPTED', 'A_REGISTERED',
       'A_SUBMITTED', 'END', 'O_ACCEPTED', 'O_CANCELLED', 'O_CREATED',
       'O_DECLINED', 'O_SELECTED', 'O_SENT', 'O_SENT_BACK',
       'W_Afhandelen leads', 'W_Beoordelen fraude', 'W_Completeren aanvraag',
       'W_Nabellen incomplete dossiers', 'W_Nabellen offertes',
       'W_Valideren aanvraag', 'W_Wijzigen contractgegevens']]
prob = df.groupby(df['last_activity']).mean().reset_index()
prob = prob.round(2)

#Check whether the probabilities sum up to 100%
check = prob.sum(axis=1)
check

prob = prob.set_index("last_activity")
prob = prob.round(2)
prob

"""## DFG"""
import graphviz

dot_all = graphviz.Digraph('state transition diagram (all edges)')

for a in prob.index:
  dot_all.node(a) # Add the activities as nodes

dot_all

THRESHOLD = 10 # %

for start_node, row in prob.iterrows(): # Loop over the dataframe rows
    for end_node, probability in row.items():  
      if probability > THRESHOLD:
        dot_all.edge(start_node, end_node, str(probability) + "%") # Add an edge to the graph

dot_all

filename=dot_all.render(filename='DFG-10% threshold')
