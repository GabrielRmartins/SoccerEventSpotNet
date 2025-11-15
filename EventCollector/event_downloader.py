import pandas as pd
import urllib.request
import os
import zipfile
import sys

def download_matches(output_path='data/matches', url="https://figshare.com/ndownloader/files/14464622"):

    matches_path_files = os.listdir(output_path)

    if len(matches_path_files) > 0:
        print("Files alredy dowloaded!")
    else:
        
        matches_file_path = os.path.join(output_path, "tmp_matches")
        matches_url = "https://figshare.com/ndownloader/files/14464622"
        
        try:
            urllib.request.urlretrieve(matches_url, matches_file_path)
        except Exception as e:
            print(f"Failed to download matches data due to error: {e}")
            
        
        if zipfile.is_zipfile(matches_file_path):
            
            with zipfile.ZipFile(matches_file_path,'r') as zip_ref:
                zip_ref.extractall(output_path)
            
            os.remove(matches_file_path)
            
            print("Matches files sucessfully downloaded!")

        
def download_teams(output_path='data/teams', url="https://figshare.com/ndownloader/files/15073697"):
 
    teams_path_files = os.listdir(output_path)

    if len(teams_path_files) > 0:
        print("Files alredy dowloaded!")
    else:
                
        teams_file_path = os.path.join(output_path, "teams.json")
        teams_url = "https://figshare.com/ndownloader/files/15073697"
        
        try:
            urllib.request.urlretrieve(teams_url, teams_file_path)
        except Exception as e:
            print(f"Failed to download teams data due to error: {e}")
                 
                
        if zipfile.is_zipfile(teams_file_path):
            
            with zipfile.ZipFile(teams_file_path,'r') as zip_ref:
                zip_ref.extractall(output_path)
            
            os.remove(teams_file_path)
            
        print("Teams files sucessfully downloaded!")


def download_events(output_path='data/events', url="https://figshare.com/ndownloader/files/14464685"):
    
    events_path_files = os.listdir(output_path)

    if len(events_path_files) > 0:
        print("Files alredy dowloaded!")
    else:
        
        events_file_path = os.path.join(output_path, "tmp_events")
        events_url = "https://figshare.com/ndownloader/files/14464685"
        
        try:
            urllib.request.urlretrieve(events_url, events_file_path)
        except Exception as e:
            print(f"Failed to download events data due to error: {e}")
            
        
        if zipfile.is_zipfile(events_file_path):
            
            with zipfile.ZipFile(events_file_path,'r') as zip_ref:
                zip_ref.extractall(output_path)
            
            os.remove(events_file_path)
            
            print("Events files sucessfully downloaded!")
    

def download_data(output_path='data'):
    
    os.makedirs(output_path,exist_ok = True)

    new_data_folders = ["matches", "teams", "events"]
    for folder in new_data_folders:
        os.makedirs(os.path.join(output_path,folder), exist_ok = True)

    download_matches(output_path=os.path.join(output_path,"matches"))
    download_teams(output_path=os.path.join(output_path,"teams"))
    download_events(output_path=os.path.join(output_path,"events"))
    
    







if __name__ == "__main__":

    if len(sys.argv)>1:
        output_path = sys.argv[1]
        download_data(output_path=output_path)
    else:
        download_data()