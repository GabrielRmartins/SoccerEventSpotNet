import pickle 
import os
import sys
from tqdm import tqdm
from utils import *
import argparse


# Generate tensors and store them in an H5 file from  event timestamps and video files

if __name__ == "__main__":

    project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Generate tensors from video files using defined segmentation strategy.")
    parser.add_argument('--output_path', type=str, default=os.path.join(project_root_path,"Common","Tensors"), help='Path to folder to save the output H5 file containing tensors.')
    parser.add_argument('--events_data_path', type=str, default=os.path.join(project_root_path,"Common", "ProcessedEvents.csv"), help='Path to the processed events data CSV file.')
    parser.add_argument('--video_info_path', type=str, default=os.path.join(project_root_path,"Common", "MatchesInfo.csv"), help='Path to the matches_info.csv file containing video information.')
    parser.add_argument('--video_data_path', type=str, default=os.path.join(project_root_path,"Common", "Videos"), help='Path to the directory containing match video files.')
    parser.add_argument('--seg_strategy', type=str, default='1', help='Segmentation strategy to extract video segments.')
    args = parser.parse_args()

    output_file = os.path.join(args.output_path,f"tensors_strategy_{args.seg_strategy}.h5")
    
    os.makedirs(args.output_path, exist_ok=True)

    if args.seg_strategy == '1':
        timestamps = collect_strategy_1_timestamps(args.events_data_path, args.video_info_path)    
    elif args.seg_strategy == '2':
        timestamps = collect_strategy_2_timestamps(args.events_data_path, args.video_info_path)    
    

    create_h5_file(output_file)

    available_videos = set(os.listdir(args.video_data_path))
    for match_id, event_id, period, start_time, end_time in tqdm(timestamps,desc="Processing events tensors"):
        if f"{match_id}.mp4" not in available_videos:
            #print(f"Video for match ID {match_id} not found. Skipping event ID {event_id}.")
            continue
        video_path = os.path.join(args.video_data_path,f"{match_id}.mp4")
        frames = extract_segment_frames(video_path, start_time, end_time)
        add_video_to_h5(output_file, event_id, frames)
    print(f"Tensors data saved to {output_file}")