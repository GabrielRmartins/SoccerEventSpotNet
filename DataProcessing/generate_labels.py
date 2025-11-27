import os
import pandas as pd
import numpy as np
import json
import sys
from sklearn.model_selection import train_test_split
import argparse

# Generate train, validation and test labels for the events based on the processed tensors H5 file
# The dataset is split into 60% train, 20% validation and 20% test
# The splitting unit is based on match IDs to avoid data leakage

def generate_labels(events_data_path, output_path, matches_split_path, exp_id): 

    os.makedirs(output_path, exist_ok=True)

    df_events = pd.read_csv(events_data_path)
    with open(matches_split_path, 'r') as f:
        split_info = json.load(f)
    train_matches = split_info['train_matches']
    val_matches = split_info['val_matches']
    test_matches = split_info['test_matches']

    if exp_id in ['1', '3', '6']:
        event_type_column = 'eventName'
    elif exp_id in ['2', '4', '7', '8', '9', '10', '11', '12']:
        event_type_column = 'subEventName'
    
    if exp_id in ['1', '2', '3', '4']:
        event_types = np.unique(df_events[event_type_column].values)
        event_to_idx = {event: idx for idx, event in enumerate(event_types)}
        with open(os.path.join(output_path, 'event_to_idx.json'), 'w') as f:
            json.dump(event_to_idx, f,indent=4)
        idx_to_event = {idx: event for event, idx in event_to_idx.items()}
        with open(os.path.join(output_path, 'idx_to_event.json'), 'w') as f:
            json.dump(idx_to_event, f,indent=4)

        train_labels = {}
        val_labels = {}
        test_labels = {}

        for index, row in df_events.iterrows():
            match_id = row["matchId"]
            event_id = row["id"]
            event_type = row[event_type_column]
            if event_type in event_types:
                label = event_to_idx[event_type]

                if match_id in train_matches:
                    train_labels[event_id] = label
                elif match_id in val_matches:
                    val_labels[event_id] = label
                elif match_id in test_matches:
                    test_labels[event_id] = label
        
        with open(os.path.join(output_path, 'train_labels.json'), 'w') as f:
            json.dump(train_labels, f,indent=4)
        
        print(f"Number of training labels generated: {len(train_labels)}")

        with open(os.path.join(output_path, 'val_labels.json'), 'w') as f:
            json.dump(val_labels, f,indent=4)

        print(f"Number of validation labels generated: {len(val_labels)}")

        with open(os.path.join(output_path, 'test_labels.json'), 'w') as f:
            json.dump(test_labels, f,indent=4) 

        print(f"Number of test labels generated: {len(test_labels)}")
            
    elif exp_id == '6':
        event_types = ['Pass', 'Duel', 'Others on the ball', 'Free Kick', 'Interruption']
        event_to_idx = {event: idx for idx, event in enumerate(event_types)}
        with open(os.path.join(output_path, 'event_to_idx.json'), 'w') as f:
            json.dump(event_to_idx, f,indent=4)
        idx_to_event = {idx: event for event, idx in event_to_idx.items()}
        with open(os.path.join(output_path, 'idx_to_event.json'), 'w') as f:
            json.dump(idx_to_event, f,indent=4)

        train_labels = {}
        val_labels = {}
        test_labels = {}

        for index, row in df_events.iterrows():
            match_id = row["matchId"]
            event_id = row["id"]
            event_type = row[event_type_column]
            if event_type in event_types:
                label = event_to_idx[event_type]

                if match_id in train_matches:
                    train_labels[event_id] = label
                elif match_id in val_matches:
                    val_labels[event_id] = label
                elif match_id in test_matches:
                    test_labels[event_id] = label
        
        with open(os.path.join(output_path, 'train_labels.json'), 'w') as f:
            json.dump(train_labels, f,indent=4)
        
        print(f"Number of training labels generated: {len(train_labels)}")

        with open(os.path.join(output_path, 'val_labels.json'), 'w') as f:
            json.dump(val_labels, f,indent=4)

        print(f"Number of validation labels generated: {len(val_labels)}")

        with open(os.path.join(output_path, 'test_labels.json'), 'w') as f:
            json.dump(test_labels, f,indent=4) 

        print(f"Number of test labels generated: {len(test_labels)}")

    else:
        event_types = ['Simple pass', 'Ground attacking duel', 'Ground defending duel', 'Touch', 'Ground loose ball duel']
        event_to_idx = {event: idx for idx, event in enumerate(event_types)}
        with open(os.path.join(output_path, 'event_to_idx.json'), 'w') as f:
            json.dump(event_to_idx, f,indent=4)
        idx_to_event = {idx: event for event, idx in event_to_idx.items()}
        with open(os.path.join(output_path, 'idx_to_event.json'), 'w') as f:
            json.dump(idx_to_event, f,indent=4)

        train_labels = {}
        val_labels = {}
        test_labels = {}

        number_of_instances = {'7':128, '8':256, '9':512, '10':1024, '11':2048, '12':4096}

        train_df = df_events[df_events['matchId'].isin(train_matches)]
        for event_type in event_types:
            event_df = train_df[train_df[event_type_column] == event_type]
            if len(event_df) > number_of_instances[exp_id]:
                event_df = event_df.sample(n=number_of_instances[exp_id], random_state=42)
            else:
                event_df = event_df
            for index,event in event_df.iterrows():
                event_id = event["id"]
                label = event_to_idx[event_type]
                train_labels[event_id] = label
                
            
        for index, row in df_events.iterrows():
            match_id = row["matchId"]
            event_id = row["id"]
            event_type = row[event_type_column]
            if event_type in event_types:
                label = event_to_idx[event_type]          
                if match_id in val_matches:
                    val_labels[event_id] = label
                elif match_id in test_matches:
                    test_labels[event_id] = label

        with open(os.path.join(output_path, 'train_labels.json'), 'w') as f:
            json.dump(train_labels, f,indent=4)
        
        print(f"Number of training labels generated: {len(train_labels)}")

        with open(os.path.join(output_path, 'val_labels.json'), 'w') as f:
            json.dump(val_labels, f,indent=4)

        print(f"Number of validation labels generated: {len(val_labels)}")

        with open(os.path.join(output_path, 'test_labels.json'), 'w') as f:
            json.dump(test_labels, f,indent=4) 

        print(f"Number of test labels generated: {len(test_labels)}")

           

def generate_waterfall_labels(events_data_path, output_path, matches_split_path):

    root_path = os.path.join(output_path, 'Root')
    print("Generating root model labels...")
    generate_labels(events_data_path, root_path, matches_split_path, exp_id='3')

    with open(os.path.join(root_path, 'event_to_idx.json'), 'r') as f:
        event_to_idx = json.load(f) 

    with open(matches_split_path, 'r') as f:
        split_info = json.load(f)  

    train_matches = split_info['train_matches']
    val_matches = split_info['val_matches']
    test_matches = split_info['test_matches']   

    test_labels = {}

    df_events = pd.read_csv(events_data_path)
    event_types = np.unique(df_events['eventName'].values)
    subevent_types = np.unique(df_events['subEventName'].values)
    subevent_to_idx = {event: idx for idx, event in enumerate(subevent_types)}
    idx_to_subevent = {idx: event for event, idx in subevent_to_idx.items()}

    with open(os.path.join(output_path, 'event_to_idx.json'), 'w') as f:
        json.dump(subevent_to_idx, f,indent=4)

    with open(os.path.join(output_path, 'idx_to_event.json'), 'w') as f:
        json.dump(idx_to_subevent, f,indent=4)

    for event_type in event_types:

        print(f"Generating labels for event type: {event_type}...")
        event_path = os.path.join(output_path, str(event_to_idx[event_type]))
        os.makedirs(event_path, exist_ok=True)

        train_labels = {}
        val_labels = {}
        

        event_df = df_events[df_events['eventName'] == event_type]
        new_event_to_idx = {event: idx for idx, event in enumerate(np.unique(event_df['subEventName']))}
        with open(os.path.join(event_path, 'event_to_idx.json'), 'w') as f:
            json.dump(new_event_to_idx, f,indent=4)
        new_idx_to_event = {idx: event for event, idx in new_event_to_idx.items()}
        with open(os.path.join(event_path, 'idx_to_event.json'), 'w') as f:
            json.dump(new_idx_to_event, f,indent=4)

        for index, row in event_df.iterrows():
            match_id = row["matchId"]
            event_id = row["id"]
            type = row["subEventName"]
            label = new_event_to_idx[type]
            if match_id in train_matches:
                train_labels[event_id] = label
            elif match_id in val_matches:
                val_labels[event_id] = label
            elif match_id in test_matches:
                test_labels[event_id] = label
        with open(os.path.join(event_path, 'train_labels.json'), 'w') as f:
            json.dump(train_labels, f,indent=4) 
        print(f"Number of training labels generated for event {event_type}: {len(train_labels)}")
        with open(os.path.join(event_path, 'val_labels.json'), 'w') as f:
            json.dump(val_labels, f,indent=4) 
        print(f"Number of validation labels generated for event {event_type}: {len(val_labels)}")
        
    with open(os.path.join(output_path, 'test_labels.json'), 'w') as f:
            json.dump(test_labels, f,indent=4) 
            print(f"Number of test labels generated : {len(test_labels)}")


        
    
if __name__ == "__main__":
    
    project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Generate train, validation and test labels for the events.")
    parser.add_argument('--events_data_path', type=str, default = os.path.join(project_root_path,"Common","ProcessedEvents.csv"),help='Path to the processed events data CSV file.')
    parser.add_argument('--output_path', type=str,default = os.path.join(project_root_path,"Common","Labels"), help='Path to save the generated labels.')
    parser.add_argument('--exp_id', type=str, default='1', help='Desired experiment ID labels come from.')
    parser.add_argument('--matches_split_path', type=str, default=os.path.join(project_root_path,"DataProcessing","MatchesSplit.json"), help='Path to the matches split JSON file.')
    args = parser.parse_args()

    output_path = os.path.join(args.output_path, f"Exp_{args.exp_id}")
    os.makedirs(output_path, exist_ok=True)
    if args.exp_id == '5':
        generate_waterfall_labels(args.events_data_path, output_path, args.matches_split_path)
    else:
        generate_labels(args.events_data_path, output_path, args.matches_split_path, args.exp_id)