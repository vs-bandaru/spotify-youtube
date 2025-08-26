from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import urllib.parse as p
import re
import os
import pickle
import csv
import pandas as pd
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
credential_path = "../api_keys/youtube_api_keys.json"
token_path = "../api_keys/token.pickle"
data_path = '../csv_files/youtube_api_data.csv'
comments_path = '../csv_files/youtube_api_comments.csv'


def youtube_authenticate():
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = credential_path
    creds = None
    # the file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time
    if os.path.exists(token_path):
        with open(token_path, "rb") as token:
            creds = pickle.load(token)
    # if there are no (valid) credentials availablle, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)
        # save the credentials for the next run
        with open(token_path, "wb") as token:
            pickle.dump(creds, token)

    return build(api_service_name, api_version, credentials=creds)


def get_video_id_list():
    video_id_dict = {}
    # the number of videos whose comments we want to scrap
    number = 10000
    df = pd.read_csv(comments_path)
    num_rows = df.shape[0]
    with open(data_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        if num_rows == 0:
            for row in reader:
                if int(row[0]) > number:
                    break
                else:
                    video_id_dict[row[0]] = row[1]
        else:
            last_row = df.tail(1)
            last_row_list = last_row.values.tolist()[0]
            last_song_id = last_row_list[1]
            for row in reader:
                if int(row[0]) <= last_song_id:
                    continue
                else:
                    video_id_dict[row[0]] = row[1]
    return video_id_dict


def get_comment_threads(youtube, video_id):
    results = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText"
    ).execute()

    # for item in results["items"]:
    #     comment = item["snippet"]["topLevelComment"]
    #     author = comment["snippet"]["authorDisplayName"]
    #     text = comment["snippet"]["textDisplay"]
    return results["items"]


def write_rowData(row_data):
    with open(comments_path, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)


def write_csv_file_header():
    header = ['id','video_num','video_id','comment_text']
    with open(comments_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data


def main():
    youtube = youtube_authenticate()
    write_csv_file_header()
    video_id_dict = get_video_id_list()
    for keyId in video_id_dict:
        video_id = video_id_dict[keyId]
        try:
            video_comment_list = get_comment_threads(youtube, video_id)
        except HttpError:
            continue
        else:
            if len(video_comment_list) == 0:
                continue
            i = 0
            while i < 5 and i < len(video_comment_list):
                id = video_id + str(i)
                item = video_comment_list[i]
                comment = item["snippet"]["topLevelComment"]
                text = comment["snippet"]["textDisplay"]
                row_data = [id, keyId, video_id, text]
                write_rowData(row_data)
                i = i + 1
                print(row_data)
    print("Successfully get comments records of 10000 videos in youtube_api_data.csv")


if __name__ == '__main__':
    main()