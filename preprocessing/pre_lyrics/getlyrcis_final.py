import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from googlesearch import search
import threading
import re


# Final version of the code to extract lyrics from the Genius website

# Function to normalize a string
def normalize_string(s):
    return re.sub(r'\W+', ' ', s)

# Function to scrape the lyrics using the Google and Genius websites
def scrape_lyrics(url, title, artist_name):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    soup = BeautifulSoup(webpage, 'html.parser')
    
    if not validate_song_info(soup, title, artist_name):
        return "Incorrect Song Information or Artist"

    lyrics_div = soup.find('div', attrs={'data-lyrics-container': 'true'})
    if lyrics_div:
        lyrics = lyrics_div.get_text(separator="\n").strip()
        return lyrics
    else:
        return "Lyrics Not Found"

def validate_song_info(soup, title, artist_name):
    song_title_element = soup.find('span', class_='SongHeaderdesktop__HiddenMask-sc-1effuo1-11')  # Genius website harder to scrape
    artist_name_element = soup.find('a', class_='HeaderArtistAndTracklistdesktop__Artist-sc-4vdeb8-1') # Genius website harder to scrape

    song_title = song_title_element.get_text().strip() if song_title_element else ''
    artist_name_on_page = artist_name_element.get_text().strip() if artist_name_element else ''

    return normalize_string(title.lower()) == normalize_string(song_title.lower()) and \
           normalize_string(artist_name.lower()) == normalize_string(artist_name_on_page.lower())

# Function to extract the lyrics using the Google search
def Extract(title, artist_name, timeout=120):
    # title = title.rstrip()

    query = "genius lyrics " + title + " " + artist_name
    url = ''
    search_completed = False

    def search_thread():
        nonlocal url, search_completed
        try:
            for j in search(query, tld="co.in", num=1, stop=1, pause=3):
                url = j
                if 'genius' in url:
                    break
        except Exception as e:
            print(f"Error during search: {e}")
        finally:
            search_completed = True

    thread = threading.Thread(target=search_thread)
    thread.start()
    thread.join(timeout=timeout)

    if search_completed and 'genius' in url:
        try:
            print(f"Extracting lyrics for '{title}' by {artist_name}...")
            lyrics = scrape_lyrics(url, title, artist_name)
            print(f"Completed lyrics for '{title}' by {artist_name}.")
            return lyrics
        except Exception as e:
            print(f"Error extracting lyrics for '{title}' by {artist_name}: {e}")
            return "Error: " + str(e)
    else:
        print(f"No results or Timeout for '{title}' by {artist_name}")
        return "Lyrics Not Found or Timeout"

# Function to extract the lyrics and save them to a CSV file
def extract_lyrics_and_save_to_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    if 'Lyrics' not in df.columns:
        df['Lyrics'] = ''

    for index, row in df.iterrows():
        if pd.isna(row['Lyrics']) or row['Lyrics'] == '':
            lyrics = Extract(row['Title'], row['Artist'])
            df.at[index, 'Lyrics'] = lyrics
            df.to_csv(output_file, index=False)

input_file_path = 'data/data_moody/MoodyLyrics.csv'
output_file_path = 'data/MoodyLyrics_lyrcis.csv'

extract_lyrics_and_save_to_csv(input_file_path, output_file_path)
