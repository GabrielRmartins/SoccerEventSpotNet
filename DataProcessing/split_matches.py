import pandas as pd
import numpy as np
import argparse
import os
import json
from sklearn.model_selection import train_test_split


def split_matches(events_data_path,output_path):
    df_events = pd.read_csv(events_data_path)
    matches = np.unique(df_events["matchId"].values)
    train_matches, temp_matches = train_test_split(matches, test_size=0.4, random_state=42)
    val_matches, test_matches = train_test_split(temp_matches, test_size=0.5, random_state=42)
    split_info = {
        'train_matches': train_matches.tolist(),
        'val_matches': val_matches.tolist(),
        'test_matches': test_matches.tolist()
    }

    print("Number of matches in each split:")
    print(f"Train: {len(train_matches)}, Validation: {len(val_matches)}, Test: {len(test_matches)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(split_info, f, indent=4)

    print(f"Match split information saved to {output_path}")


if __name__ == "__main__":
    
    project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Split matches into train, validation and test sets.")
    parser.add_argument('--events_data_path', type=str, default=os.path.join(project_root_path,"Common", "ProcessedEvents.csv"), help='Path to the processed events data CSV file.')
    parser.add_argument('--output_path', type=str, default=os.path.join(project_root_path,"DataProcessing","MatchesSplit.json"), help='Path to save the split match information.')    
    args = parser.parse_args()

    split_matches(args.events_data_path, args.output_path)