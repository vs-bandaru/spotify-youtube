from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import urllib.parse as p
import re
import os
import pickle
import csv
import pandas as pd

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]


# get authentication
def youtube_authenticate():
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = "../api_keys/youtube_api_keys.json"
    creds = None
    # the file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time
    if os.path.exists("../api_keys/token.pickle"):
        with open("../api_keys/token.pickle", "rb") as token:
            creds = pickle.load(token)
    # if there are no (valid) credentials availablle, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)
        # save the credentials for the next run
        with open("../api_keys/token.pickle", "wb") as token:
            pickle.dump(creds, token)

    return build(api_service_name, api_version, credentials=creds)


def get_header_info():
    global index_url
    df = pd.read_csv('../csv_files/spotify_youtube.csv')
    columns = df.columns.tolist()
    for i in range(len(columns)):
        if columns[i] == "Url_youtube":
            index_url = i
            break
    return index_url


# get a list of url from .csv file
def get_csv_url(index_url):
    id_video_dict = {}
    # the number of data scraped
    number = 100
    # judge csv file data
    df = pd.read_csv('../csv_files/youtube_api_data.csv')
    num_rows = df.shape[0]
    with open('../csv_files/spotify_youtube.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        if num_rows == 0:
            for row in reader:
                if int(row[0]) > number:
                    break
                else:
                    id_video_dict[row[0]] = row[index_url]
        else:
            last_row = df.tail(1)
            last_row_list = last_row.values.tolist()[0]
            last_song_id = last_row_list[0]
            for row in reader:
                if int(row[0]) <= last_song_id:
                    continue
                if int(row[0]) > number:
                    break
                else:
                    id_video_dict[row[0]] = row[index_url]
    return id_video_dict


# write features of songs from youtube into .csv file
def write_csv_file_header():
    header = ['id', 'video_id', 'title', 'channel_title', 'description', 'comment_count', 'like_count', 'view_count']
    with open('../csv_files/youtube_api_data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        # writer.writerows(data)


# get video_id by parsing url
def get_video_id_by_url(url):
    """
    Return the Video ID from the video `url`
    """
    # split URL parts
    parsed_url = p.urlparse(url)
    # get the video ID by parsing the query of the URL
    video_id = p.parse_qs(parsed_url.query).get("v")
    if video_id:
        return video_id[0]
    else:
        raise Exception(f"Wasn't able to parse video URL: {url}")


# request execute
def get_video_details(youtube, **kwargs):
    return youtube.videos().list(
        part="snippet,contentDetails,statistics",
        **kwargs
    ).execute()


def print_video_infos(video_response, video_id, keyId):
    global id, videoId, statistics, content_details, channel_title, title, description, publish_time, comment_count, like_count, view_count
    videoId = video_id
    id = keyId
    try:
        if len(video_response.get("items")) == 0:
            return
        items = video_response.get("items")[0]
        # get the snippet, statistics & content details from the video response
        snippet = items["snippet"]
        statistics = items["statistics"]
        content_details = items["contentDetails"]
        # get infos from the snippet
        channel_title = snippet["channelTitle"]
        title = snippet["title"]
        description = snippet["description"]
        publish_time = snippet["publishedAt"]
        # get stats infos
        if 'commentCount' in statistics:
            comment_count = statistics["commentCount"]
        else:
            comment_count = 0
        if 'likeCount' in statistics:
            like_count = statistics["likeCount"]
        else:
            like_count = 0
        if 'viewCount' in statistics:
            view_count = statistics["viewCount"]
        else:
            view_count = 0
    except KeyError:
        raise Exception(f"keyerror: {statistics}")
    else:
        row_data = [id, videoId, title, channel_title, description, comment_count, like_count, view_count]
        print(row_data)
    return row_data


def write_rowData(row_data):
    with open('../csv_files/youtube_api_data.csv', 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)



def main():
    # authenticate to YouTube API
    youtube = youtube_authenticate()
    index_url = get_header_info()
    write_csv_file_header()
    id_video_dict = get_csv_url(index_url)
    # get video_id into dict
    for keyId in list(id_video_dict.keys()):
        video_url = id_video_dict[keyId]
        if video_url == '':
            id_video_dict.pop(keyId)
            continue
        video_id = get_video_id_by_url(video_url)
        if video_id == '':
            id_video_dict.pop(keyId)
            continue
        id_video_dict[keyId] = video_id
    for keyId in id_video_dict:
        video_id = id_video_dict[keyId]
        # make API call to get video info
        response = get_video_details(youtube, id=video_id)
        # print extracted video infos
        row_data = print_video_infos(response, video_id, keyId)
        # data.append(row_data)
        if row_data == None:
            continue
        else:
            write_rowData(row_data)
    print("10000 records written to youtube_api_data.csv")


if __name__ == '__main__':
    main()

