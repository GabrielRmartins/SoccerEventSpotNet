import os
import h5py
import pandas as pd
import numpy as np
import json
import sys
from sklearn.model_selection import train_test_split

# Generate train, validation and test labels for the events based on the processed tensors H5 file
# The dataset is split into 60% train, 20% validation and 20% test
# The splitting unit is based on match IDs to avoid data leakage

def generate_labels(events_data_path, output_path, event_type_column='eventName'):
    df_events = pd.read_csv(events_data_path)
    matches = np.unique(df_events["matchId"].values)
    train_matches, temp_matches = train_test_split(matches, test_size=0.4, random_state=42)
    val_matches, test_matches = train_test_split(temp_matches, test_size=0.5, random_state=42)
    event_types = np.unique(df_events[event_type_column].values)
    event_to_idx = {event: idx for idx, event in enumerate(event_types)}
    with open(os.path.join(output_path, 'event_to_idx.json'), 'w') as f:
        json.dump(event_to_idx, f,indent=4)

    train_labels = {}
    val_labels = {}
    test_labels = {}

    for index, row in df_events.iterrows():
        match_id = row["matchId"]
        event_id = row["id"]
        event_type = row[event_type_column]
        label = event_to_idx[event_type]

        if match_id in train_matches:
            train_labels[event_id] = label
        elif match_id in val_matches:
            val_labels[event_id] = label
        elif match_id in test_matches:
            test_labels[event_id] = label
    
    with open(os.path.join(output_path, 'train_labels.json'), 'w') as f:
        json.dump(train_labels, f,indent=4)
    with open(os.path.join(output_path, 'val_labels.json'), 'w') as f:
        json.dump(val_labels, f,indent=4)
    with open(os.path.join(output_path, 'test_labels.json'), 'w') as f:
        json.dump(test_labels, f,indent=4)  

    
if __name__ == "__main__":
    project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    events_data_path = os.path.join(project_root_path,"Common","processed_events.csv")
    output_path = os.path.join(project_root_path,"DataProcessing","labels")
    if len(sys.argv) > 1:
        events_data_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    os.makedirs(output_path, exist_ok=True)
    generate_labels(events_data_path, output_path)