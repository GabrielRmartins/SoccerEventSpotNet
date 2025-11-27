import pandas as pd
import numpy as np
import os
import sys
import argparse

# Filter and process event data based on available match videos on match_info.csv

def process_event_data(events_data_path, video_info_path, output_path):
    # Load events and video info data
    df_video_info = pd.read_csv(video_info_path)

    collected_matches_ids = np.unique(df_video_info["matchId"].values)

    print("Number of collected matches with videos found:", len(collected_matches_ids))

    events_dfs = []

    for event_df in os.listdir(events_data_path):
        event_file_path = os.path.join(events_data_path, event_df)
        df = pd.read_json(event_file_path, orient= 'records')
        target_df = df[df['matchId'].isin(collected_matches_ids)]
        events_dfs.append(target_df)

    target_events_df = pd.concat(events_dfs)

    print("Number of events after filtering for collected matches:", len(target_events_df))    
    target_events_df['subEventName'] = target_events_df['subEventName'].replace('','Offside')
    target_events_df.to_csv(output_path, index=False)
    print(f"Processed events data saved to {output_path}")
    
    

if __name__ == "__main__":

    project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Process event data based on collected events and videos.")
    parser.add_argument('--events_data_path', type=str, default = os.path.join(project_root_path,"Common","WyscoutTop5","events"), help='Path to the directory containing event data files.')
    parser.add_argument('--video_info_path', type=str, default = os.path.join(project_root_path,"Common","MatchesInfo.csv"), help='Path to the matches_info.csv file containing video information.')
    parser.add_argument('--output_path', type=str, default = os.path.join(project_root_path,"Common","ProcessedEvents.csv") , help='Path to save the processed events data.')

    args = parser.parse_args()

    process_event_data(args.events_data_path, args.video_info_path, args.output_path)