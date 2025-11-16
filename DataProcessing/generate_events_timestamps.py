import pandas as pd
import json
import os
import sys
import pickle

# Generate event timestamps with defined end times for all events in the processed events data

from utils import collect_all_timestamps

def define_events_endtimes(events_data_path):
    df_events = pd.read_csv(events_data_path)
    df_events = df_events.sort_values(by=['matchId','matchPeriod','eventSec'],ascending=True).reset_index(drop=True).copy()
    df_events['eventSecEnd'] = df_events['eventSec'].shift(-1)
    df_events.loc[df_events['eventSecEnd'] < df_events['eventSec'], 'eventSecEnd'] = df_events.loc[df_events['eventSecEnd'] < df_events['eventSec'], 'eventSec'] + 1.5
    return df_events

def generate_timestamps (events_data_path, video_data_path, output_path):
    df_events = define_events_endtimes(events_data_path)
    timestamps = collect_all_timestamps(df_events, video_data_path)
    output_file = os.path.join(output_path, "timestamps.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(timestamps, f)
    print(f"Timestamps data saved to {output_file}")

if __name__ == "__main__":
    project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    events_data_path = os.path.join(project_root_path,"Common","processed_events.csv")
    video_data_path = os.path.join(project_root_path,"Common","matches_info.csv")
    output_path = os.path.join(project_root_path,"DataProcessing","timestamps")
    os.makedirs(output_path, exist_ok=True)
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    if len(sys.argv) > 2:
        events_data_path = sys.argv[2]
    if len(sys.argv) > 3:
        video_data_path = sys.argv[3]

    generate_timestamps (events_data_path, video_data_path, output_path)



    
        
    