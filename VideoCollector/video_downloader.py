from yt_dlp import YoutubeDL
import os
import pandas as pd
import sys
import json

def download_video(url,match_id, output_path='match_videos', options_path = 'download_options.json'):
    """
    Downloads youtube video from url and saves it with name pattern "<match_id>.mp4" in best video and audio qualities available in mp4 extension.

    The default video saving path is match_videos.

    For better project data managment match_id should be the game identification number that appears on its event data.
    
    """
    os.makedirs(output_path,exist_ok = True)

    with open(options_path,'r') as f:
        ydl_opts = json.load(f)

    ydl_opts['outtmpl'] = os.path.join(output_path, f'{match_id}.mp4')

    try:
        with YoutubeDL(ydl_opts) as ydl:
            # Download content
            ydl.download([url])
            print(f"\nDownloaded ended successfully, video file saved on: {output_path}")
    except Exception as e:
        print(f"An unexpected error raised during video collect: {str(e)}")



def download_all_videos(output_path='match_videos', options_path = 'download_options.json'):
    """
    Downloads all videos from a csv file containing video links and matches ids.

    The csv file should have at least two columns: 'Game_Id' and 'Link', where 'matchId' is the unique identifier for each game and 'videoLink' is the URL to the video.

    """
    project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root_path,'Common','matches_info.csv')

    if len(sys.argv)>1:
        output_path = sys.argv[1]

    if len(sys.argv)>2:
        input_path = sys.argv[2]

    if len(sys.argv)>3:
        options_path = sys.argv[3]

    try:
        df = pd.read_csv(input_path)
        print(f"Video list loaded successfully, {len(df)} videos added to download queue!")
    except Exception as e:
        print(f"An error occurred while loading video list: {str(e)}")
        exit(0)
    

    
    if df.empty:
        print("The video list is empty. Please add URLs to video list file.")
        exit(0)
        
    
    num_videos = len(df)
    video_iterator = 1
    df_video_links = df.loc[:,['matchId','videoLink']].copy()
    for _,row in df_video_links.iterrows():
        match_id = row['matchId']
        link = row['videoLink']
        print(f"Downloading video {video_iterator}/{num_videos} from: {link}\n")
        download_video(link,match_id, output_path=output_path,options_path=options_path)
        video_iterator += 1


if __name__ == "__main__":

    download_all_videos()
