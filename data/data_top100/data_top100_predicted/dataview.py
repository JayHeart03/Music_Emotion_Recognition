import matplotlib.pyplot as plt
import pandas as pd

# Dictionary to store emotion counts per year
emotion_counts_per_year = {}

# Looping through each year from 2013 to 2023
for year in range(2013, 2024):
    # Constructing the file path for the current year's data
    file_path = f'data/data_top100/data_top100_predicted/{year}_predictions.csv'
    
    data = pd.read_csv(file_path)

    # Counting and normalizing the frequency of each predicted mood/emotion in the dataset
    emotion_counts = data['predicted_mood'].value_counts(normalize=True) * 100  # Convert to percentage
    emotion_counts_per_year[year] = emotion_counts

emotion_df = pd.DataFrame(emotion_counts_per_year).fillna(0)

plt.figure(figsize=(15, 8))

for emotion in emotion_df.index:
    plt.plot(emotion_df.columns, emotion_df.loc[emotion], marker='o', label=emotion)

plt.title('Emotion Trends from 2013 to 2023')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend(title='Emotion', loc='upper left')
plt.xticks(range(2013, 2024))  # Setting the x-axis ticks to the years
plt.grid(True)  

plt.show()
