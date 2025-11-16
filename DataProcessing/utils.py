import pandas as pd
import cv2
import numpy as np
import h5py

def collect_match_timestamps(match_id,events_data_path, video_data_path):

    # Collect all event timestamps for a given matchId

    df_events = pd.read_csv(events_data_path)
    df_video_info = pd.read_csv(video_data_path)
    df_match_events = df_events[df_events["matchId"] == int(match_id)]
    video_info = df_video_info[df_video_info["matchId"] == int(match_id)]
    first_half_start = video_info["1st_half_start"].values[0]
    second_half_start = video_info["2nd_half_start"].values[0]
    timestamps = []
    for index,row in df_match_events.iterrows():
        event_id = row["id"]
        period = row["matchPeriod"]
        if period == "1H":
            start_timestamp = first_half_start + row["eventSec"]
            end_timestamp = first_half_start + row["eventSecEnd"] 
            
        elif period == "2H":
            start_timestamp = second_half_start + row["eventSec"]
            end_timestamp = second_half_start + row["eventSecEnd"] 
            
        else:
            continue
        timestamps.append((event_id,period,start_timestamp,end_timestamp))
    return timestamps

def collect_all_timestamps(df_events, video_info_path):

    # Collect all event timestamps for all matches given an events dataframe and video_info dataframe

    df_video_info = pd.read_csv(video_info_path)
    matches = np.unique(df_events["matchId"].values)
    timestamps = []
    for match_id in matches:
        df_match_events = df_events[df_events["matchId"] == int(match_id)]
        video_info = df_video_info[df_video_info["matchId"] == int(match_id)]
        first_half_start = video_info["1st_half_start"].values[0]
        second_half_start = video_info["2nd_half_start"].values[0]
        
        for index,row in df_match_events.iterrows():
            event_id = row["id"]
            period = row["matchPeriod"]
            if period == "1H":
                start_timestamp = first_half_start + row["eventSec"]
                end_timestamp = first_half_start + row["eventSecEnd"] 
                
            elif period == "2H":
                start_timestamp = second_half_start + row["eventSec"]
                end_timestamp = second_half_start + row["eventSecEnd"] 
                
            else:
                continue
            timestamps.append((match_id,event_id,period,start_timestamp,end_timestamp))
    return timestamps


def extract_segment_frames(video_path, start_time, end_time, num_frames=13, size=(252, 252)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Unable to open video at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    end_frame = min(end_frame, total_frames - 1)

    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        frames.append(frame)

    cap.release()

    return np.array(frames, dtype=np.uint8)


def create_h5_file(h5_path):
    with h5py.File(h5_path, 'w') as f:
        f.create_group("data")
    print(f"File {h5_path} created successfully.")


def add_video_to_h5(h5_path, video_id, video_data):
    
    with h5py.File(h5_path, 'a') as f:
        data_group = f["data"]
        key = str(video_id)

        
        if key in data_group:
            
            del data_group[key]  

        data_group.create_dataset(
            key,
            data=video_data.astype('uint8'),
            compression="gzip",
            compression_opts=4
        )


    
