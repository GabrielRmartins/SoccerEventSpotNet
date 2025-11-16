import pandas as pd
import numpy as np
import os
import sys

# Filter and process event data based on available match videos on match_info.csv

def process_event_data(events_data_path, video_info_path):
    # Load events and video info data
    df_video_info = pd.read_csv(video_info_path)

    collected_matches_ids = np.unique(df_video_info["matchId"].values)

    events_dfs = []

    for event_df in os.listdir(events_data_path):
        event_file_path = os.path.join(events_data_path, event_df)
        df = pd.read_json(event_file_path, orient= 'records')
        target_df = df[df['matchId'].isin(collected_matches_ids)]
        events_dfs.append(target_df)

    target_events_df = pd.concat(events_dfs)


    project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    output_events_path = os.path.join(project_root_path,"Common","processed_events.csv")

    target_events_df.to_csv(output_events_path, index=False)
    print(f"Processed events data saved to {output_events_path}")
    
    

if __name__ == "__main__":

    if len(sys.argv) >=3:
        events_data_path = sys.argv[1]
        video_info_path = sys.argv[2]     
        process_event_data(events_data_path, video_info_path)

    elif len(sys.argv) >= 2:
        events_data_path = sys.argv[1]
        project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_info_path = os.path.join(project_root_path,"Common","matches_info.csv")
        process_event_data(events_data_path,video_info_path)

    else:
        project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        events_data_path = os.path.join(project_root_path,"EventCollector","data","events")
        video_info_path = os.path.join(project_root_path,"Common","matches_info.csv")
        process_event_data(events_data_path, video_info_path)
        