import urllib.request
import os
import zipfile
import sys
import argparse

def download_matches(output_path, url):

    matches_path_files = os.listdir(output_path)

    if len(matches_path_files) > 0:
        print("Files alredy dowloaded!")
    else:
        
        matches_file_path = os.path.join(output_path, "tmp_matches")
        matches_url = url
        
        try:
            urllib.request.urlretrieve(matches_url, matches_file_path)
        except Exception as e:
            print(f"Failed to download matches data due to error: {e}")
            
        
        if zipfile.is_zipfile(matches_file_path):
            
            with zipfile.ZipFile(matches_file_path,'r') as zip_ref:
                zip_ref.extractall(output_path)
            
            os.remove(matches_file_path)
            
            print("Matches files sucessfully downloaded!")

        
def download_teams(output_path, url):
 
    teams_path_files = os.listdir(output_path)

    if len(teams_path_files) > 0:
        print("Files alredy dowloaded!")
    else:
                
        teams_file_path = os.path.join(output_path, "teams.json")
        teams_url = url
        
        try:
            urllib.request.urlretrieve(teams_url, teams_file_path)
        except Exception as e:
            print(f"Failed to download teams data due to error: {e}")
                 
                
        if zipfile.is_zipfile(teams_file_path):
            
            with zipfile.ZipFile(teams_file_path,'r') as zip_ref:
                zip_ref.extractall(output_path)
            
            os.remove(teams_file_path)
            
        print("Teams files sucessfully downloaded!")


def download_events(output_path, url):
    
    events_path_files = os.listdir(output_path)

    if len(events_path_files) > 0:
        print("Files alredy dowloaded!")
    else:
        
        events_file_path = os.path.join(output_path, "tmp_events")
        events_url = url
        
        try:
            urllib.request.urlretrieve(events_url, events_file_path)
        except Exception as e:
            print(f"Failed to download events data due to error: {e}")
            
        
        if zipfile.is_zipfile(events_file_path):
            
            with zipfile.ZipFile(events_file_path,'r') as zip_ref:
                zip_ref.extractall(output_path)
            
            os.remove(events_file_path)
            
            print("Events files sucessfully downloaded!")
    

def download_data():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Download soccer event, match, and team data.')
    parser.add_argument('--output_path', type=str, default=os.path.join(project_root,"Common","WyscoutTop5"), help='Directory path to save the downloaded data.') 
    parser.add_argument('--url_events', type=str, default="https://figshare.com/ndownloader/files/14464685", help='URL to download events from (if different from Wyscout Top 5 Figshare Dataset).')
    parser.add_argument('--url_matches', type=str, default="https://figshare.com/ndownloader/files/14464622", help='URL to download matches from (if different from Wyscout Top 5 Figshare Dataset).')
    parser.add_argument('--url_teams', type=str, default="https://figshare.com/ndownloader/files/15073697", help='URL to download teams from (if different from Wyscout Top 5 Figshare Dataset).')
    args = parser.parse_args()
    
    os.makedirs(args.output_path,exist_ok = True)

    new_data_folders = ["matches", "teams", "events"]
    for folder in new_data_folders:
        os.makedirs(os.path.join(args.output_path,folder), exist_ok = True)

    download_matches(output_path=os.path.join(args.output_path,"matches"),url=args.url_matches)
    download_teams(output_path=os.path.join(args.output_path,"teams"),url=args.url_teams)
    download_events(output_path=os.path.join(args.output_path,"events"),url=args.url_events)
    
    


if __name__ == "__main__":

    download_data()