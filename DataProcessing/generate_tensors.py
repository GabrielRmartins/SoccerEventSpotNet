import pickle 
import os
import sys
from tqdm import tqdm
from utils import extract_segment_frames,create_h5_file,add_video_to_h5

# Generate tensors and store them in an H5 file from  event timestamps and video files

if __name__ == "__main__":
    project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_data_path = os.path.join(project_root_path,"VideoCollector","match_videos")
    timestamps_path = os.path.join(project_root_path,"DataProcessing","timestamps","timestamps.pkl")
    output_file = os.path.join(project_root_path,"DataProcessing","processed_tensors","tensors.h5")
    
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    if len(sys.argv) > 2:
        video_data_path = sys.argv[2]
    if len(sys.argv) > 3:
        timestamps_path = sys.argv[3]

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    with open(timestamps_path, 'rb') as f:
        timestamps = pickle.load(f)

    create_h5_file(output_file)

    available_videos = set(os.listdir(video_data_path))
    for match_id, event_id, period, start_time, end_time in tqdm(timestamps,desc="Processing events tensors"):
        if f"{match_id}.mp4" not in available_videos:
            #print(f"Video for match ID {match_id} not found. Skipping event ID {event_id}.")
            continue
        video_path = os.path.join(video_data_path,f"{match_id}.mp4")
        frames = extract_segment_frames(video_path, start_time, end_time)
        add_video_to_h5(output_file, event_id, frames)
    print(f"Tensors data saved to {output_file}")