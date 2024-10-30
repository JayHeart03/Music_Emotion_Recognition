import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv('data/MoodyLyrics.csv')
text = ' '.join(df['lyrics'])    

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

plt.show()
