# %%

import csv



# %%


# %%
cry_loc = "dataClassified/cry_data"
scream_loc = "dataClassified/scream_data"
control_loc = "dataClassified/control_data"

# %%
import os

# %% [markdown]
# # CRY: 1
# # Scream: 0
# # Control: 2

# %%
# Save a csvfile in this format:
# file_path, label inside dataClassified folder

def load_data(loc, id):
    data = []
    for filename in os.listdir(loc):
        if filename.endswith(".wav"):
            if id == 1:
                label = "cry_data"
            
            elif id == 0:
                label = "scream_data"
            
            else:
                label = "control_data"
            data.append((label + "/" + filename, id))
    return data

cry_data = load_data(cry_loc, 1)
scream_data = load_data(scream_loc, 0)
control_data = load_data(control_loc, 2)

# %%
data = cry_data + scream_data + control_data

# Shuffle the data
import random
random.seed(42)
random.shuffle(data)



# %%
# Save the data to a csv file
csv_loc = "dataClassified/data.csv"
with open(csv_loc, mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    data_writer.writerow(['file_path', 'label'])
    for d in data:
        data_writer.writerow([d[0], d[1]])


# %%



