import os
import pandas as pd
from collections import OrderedDict
import youtube_dl
from multiprocessing import Pool
from functools import partial
import time
import glob
import pytube

def download_save_video_tube(video_id, class_name, save_path):
    
    with youtube_dl.YoutubeDL() as ydl:
            dictMeta = ydl.extract_info(
                'https://www.youtube.com/watch?v='+video_id,
                download=False # We just want to extract the info
            )
    iDuration = dictMeta["duration"]
    
    try:
        # if the video is less than 13 minutes = 780 seconds
        if iDuration < 780:
            url = 'https://www.youtube.com/watch?v='+video_id
            youtube = pytube.YouTube(url)

            video = youtube.streams.get_highest_resolution()
            video.download(save_path, filename=video_id+".mp4")
        else:
            # too long to download
            pass
        
    except:
        print("Class: {} - Video: {} --> Can NOT be downloaded!...".format(class_name, video_id))
        print("----------------------------------------------------------------------------\n")    
        
        
def download_save_video(video_id, class_name, save_path):
        
        
    ydl_opts = {
        'format': 'bestvideo/best',
        'outtmpl': save_path+'/%(id)s.mp4',   
    }
    
    
    
    try:
        with youtube_dl.YoutubeDL() as ydl:
            dictMeta = ydl.extract_info(
                'https://www.youtube.com/watch?v='+video_id,
                download=False # We just want to extract the info
            )
        iDuration = dictMeta["duration"]
        
        # if the video is less than 8 minutes = 480 seconds
        if iDuration < 480:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download(['https://www.youtube.com/watch?v='+video_id])
        else:
            # too long to download
            pass
        
    except:
        print("Class: {} - Video: {} --> Can NOT be downloaded!...".format(class_name, video_id))
        print("----------------------------------------------------------------------------\n")

def parse_CSV(input_csv):

    # takes path as an input
    df = pd.read_csv(input_csv)
    #print(df.columns)
    df_classes = list(set(df["action_name"]))
    
    df_info = dict()
    
    for df_class in df_classes:
        video_urls = df[df["action_name"] == df_class]["youtube_id"].values.tolist()
        
        df_info[df_class] = video_urls
    
    return df_info
        
if __name__ == '__main__':
    
    # constants
    input_df_path = "C:/Users/PC/Desktop/FewShotPhd/data_hvu/hvu_classes/train_df.csv"
    save_path = "D:/HolisticVideoUnderstanding/train"
    
    input_df = parse_CSV(input_df_path)
    
    classes = input_df.keys()
    
    # number of processes
    p = Pool(200)
    
    start_time = time.time()
    for action_class in classes:
        
        # check the save path
        save_folder = os.path.join(save_path, action_class) 
        if not os.path.exists(save_folder):
            print("Working: {}".format(save_folder))
            os.makedirs(save_folder)
            
            video_urls = input_df[action_class]
            
            save_constants=partial(download_save_video_tube, class_name=action_class, save_path=save_folder) 
            p.map(save_constants, video_urls)
        
        else:
            print("Passed: {}".format(save_folder))
            continue    
    print("--- %s seconds ---" % (time.time() - start_time))
    

    
    