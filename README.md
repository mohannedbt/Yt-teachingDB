# Why AI Can't Teach Well
## In the Pursuit of More Optimal Ways to Explain Things

---

### I) Introduction
This cell presents an example of an AI explanation of **FFT (Fast Fourier Transform)**.  

> These results are generated using **the free version of ChatGPT (gpt-5o-mini)**.

![image.png](README_files/4faad0bb-b785-4018-af50-286b78f19a6d.png)

From the screenshot, it is evident that AI explanations can sometimes be suboptimal. Observed issues include:

- Use of terms like $O(N)$ `complexity` without clarifying if the user understands **Big-O notation**.
- Explanations are often **unintuitive** and **not concise** for a brief explanation.
- The explanation begins with a **definition** rather than a **problem-oriented approach**.
- Lacks **interactivity** that could engage the learner more effectively.

**Question:**  
If AI had the capability to provide **animated, problem-based explanations**, would the learning experience improve?

---

### II) Premise
The premise of this project is to **analyze and collect data for eventual AI training**.  

- The dataset focuses on content from **popular YouTube channels and websites** that specialize in explaining **complex topics**.
- These sources have established **reputation and high engagement**, providing a model for effective teaching strategies.
- The goal is to analyze **language use, phrasing, and problem framing** to inform **future AI fine-tuning**.
### III)Primal exploration
the code in `Scraping.py` generates a csv file with the current date in the `DataSet` directory
we will analyse in this paragraph the Scraped urls
#### 1) Surface-level Examination


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df=pd.read_csv("DataSet/Scraped Videos/Scraped2025-11-15.csv")
print(df.dtypes)
```

    video_id             object
    title                object
    channel              object
    duration             object
    duration_seconds    float64
    views                 int64
    likes                 int64
    topics               object
    url                  object
    search_term          object
    dtype: object
    

As you can see the head of the CSV contains the Above info,we will focus our analysis on the following things :
- Title
- channel
- duration
- views
- likes
- search_term

Our bases of comparison at this stage will be the **duration_seconds** ,the **title** and the **Views** and our Target is visualizing the **likes**
> the likes at this stage is our **best indicator** bcz the **"True" engagement** info is relatively **low and exclusive**

##### let's start by the easiest "the **Duration_seconds** and the **Views**"


```python
#Selecting the numerical columns
df_num=df.loc[:,["views","likes","duration_seconds"]]
#plotting the numerical columns in a histogram
for col in df_num.columns:
    plt.figure()
    sns.distplot(df[col])

```

    C:\Users\mohan\AppData\Local\Temp\ipykernel_2064\3485866643.py:6: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(df[col])
    C:\Users\mohan\AppData\Local\Temp\ipykernel_2064\3485866643.py:6: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(df[col])
    C:\Users\mohan\AppData\Local\Temp\ipykernel_2064\3485866643.py:6: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(df[col])
    


    
![png](README_files/README_3_1.png)
    



    
![png](README_files/README_3_2.png)
    



    
![png](README_files/README_3_3.png)
    


As you can see by the initial visualisation the numbers don't seem centered in the middle. I think the reason is we're visualizing by $1e7$ unit
- an idea is to **visualize the Mean** with a line


```python
import seaborn as sns
import matplotlib.pyplot as plt

df_num = df[["views", "likes", "duration_seconds"]]

for col in df_num.columns:
    plt.figure()
    sns.histplot(df_num[col], kde=True)
    mean_val=df_num[col].mean()
    plt.axvline(mean_val, color='red', linestyle='--', label='Mean')
     # Add text label next to the vertical line
    plt.text(mean_val, plt.ylim()[1]*0.8, f"{mean_val:.2f}", 
             color='red', rotation=0, va='center')
    plt.legend()
    plt.title(col)

```


    
![png](README_files/README_5_0.png)
    



    
![png](README_files/README_5_1.png)
    



    
![png](README_files/README_5_2.png)
    


The Mean is **Left scewed** so the best way to more visualize it is by using the $log() \;Scale$


```python
for col in df_num.columns:
    plt.figure()
    sns.histplot(df_num[col].apply(lambda x: np.log1p(x)), kde=True)
    plt.title(col + " (log-transformed)")
```


    
![png](README_files/README_7_0.png)
    



    
![png](README_files/README_7_1.png)
    



    
![png](README_files/README_7_2.png)
    


You can see now that the shapes of the functions are more like **Normal Distributions**. 
Now we will check for correlation between these differents parameters


```python
plt.figure(figsize=(6,4))
corr = df_num.corr()  # Compute correlation matrix

sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()
```


    
![png](README_files/README_9_0.png)
    


So from the Correlation Matrix There is no direct correlation between **duration_seconds** and the number of **views/likes** but this pair( **views an likes**) seem very correlated

##### Title Analysis
we will dive in this part in title analysis to do that we must tokenise it


```python
import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove punctuation
    return text.split()

df["title_tokens"] = df["title"].apply(preprocess)
```

After tokenizing it to differents words we can see our title_tokens and do operations on them


```python
####Removing stop words:
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

df["title_tokens"] = df["title_tokens"].apply(
    lambda tokens: [w for w in tokens if w not in stop_words]
)

```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\mohan\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
###Counting words
from collections import Counter

word_counts = Counter()

for tokens in df["title_tokens"]:
    word_counts.update(tokens)

word_counts_df = pd.DataFrame(word_counts.items(), columns=["word", "count"])
word_counts_df = word_counts_df.sort_values(by="count", ascending=False)
word_counts_df.head(20)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>174</th>
      <td>complex</td>
      <td>135</td>
    </tr>
    <tr>
      <th>175</th>
      <td>analysis</td>
      <td>97</td>
    </tr>
    <tr>
      <th>2</th>
      <td>derivatives</td>
      <td>56</td>
    </tr>
    <tr>
      <th>5</th>
      <td>calculus</td>
      <td>44</td>
    </tr>
    <tr>
      <th>10</th>
      <td>explained</td>
      <td>25</td>
    </tr>
    <tr>
      <th>0</th>
      <td>derivative</td>
      <td>22</td>
    </tr>
    <tr>
      <th>35</th>
      <td>functions</td>
      <td>22</td>
    </tr>
    <tr>
      <th>61</th>
      <td>rule</td>
      <td>21</td>
    </tr>
    <tr>
      <th>22</th>
      <td>minutes</td>
      <td>18</td>
    </tr>
    <tr>
      <th>181</th>
      <td>numbers</td>
      <td>18</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>18</td>
    </tr>
    <tr>
      <th>38</th>
      <td>math</td>
      <td>16</td>
    </tr>
    <tr>
      <th>23</th>
      <td>basic</td>
      <td>16</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2</td>
      <td>15</td>
    </tr>
    <tr>
      <th>172</th>
      <td>3</td>
      <td>15</td>
    </tr>
    <tr>
      <th>178</th>
      <td>analytic</td>
      <td>13</td>
    </tr>
    <tr>
      <th>26</th>
      <td>differentiation</td>
      <td>12</td>
    </tr>
    <tr>
      <th>135</th>
      <td>part</td>
      <td>12</td>
    </tr>
    <tr>
      <th>158</th>
      <td>lecture</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>introduction</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



As you can see the most used things are related to our theme (in this case complexe analysis and derivatives)
but there is outliers like **Explained,introduction,minutes**
these words seem to be the one attracting users to the videos


```python
### analysing title tokens in function of views
from collections import defaultdict

word_views = defaultdict(list)

for i, row in df.iterrows():
    for word in row['title_tokens']:
        word_views[word].append(row['views'])

# Compute average views per word
avg_views_per_word = {word: sum(views)/len(views) for word, views in word_views.items()}

# Convert to DataFrame and sort
avg_views_df = pd.DataFrame(avg_views_per_word.items(), columns=['word', 'avg_views'])
avg_views_df = avg_views_df.sort_values(by='avg_views', ascending=False)
avg_views_df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>avg_views</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>376</th>
      <td>fourier</td>
      <td>1.854372e+07</td>
    </tr>
    <tr>
      <th>377</th>
      <td>heat</td>
      <td>1.854372e+07</td>
    </tr>
    <tr>
      <th>380</th>
      <td>de4</td>
      <td>1.854372e+07</td>
    </tr>
    <tr>
      <th>378</th>
      <td>drawing</td>
      <td>1.854372e+07</td>
    </tr>
    <tr>
      <th>379</th>
      <td>circles</td>
      <td>1.854372e+07</td>
    </tr>
    <tr>
      <th>428</th>
      <td>map</td>
      <td>1.512006e+07</td>
    </tr>
    <tr>
      <th>312</th>
      <td>flow</td>
      <td>9.275553e+06</td>
    </tr>
    <tr>
      <th>240</th>
      <td>real</td>
      <td>8.478478e+06</td>
    </tr>
    <tr>
      <th>239</th>
      <td>imaginary</td>
      <td>8.478478e+06</td>
    </tr>
    <tr>
      <th>273</th>
      <td>hypothesis</td>
      <td>6.105067e+06</td>
    </tr>
    <tr>
      <th>417</th>
      <td>numberphile</td>
      <td>5.836491e+06</td>
    </tr>
    <tr>
      <th>237</th>
      <td>visualizing</td>
      <td>5.060560e+06</td>
    </tr>
    <tr>
      <th>47</th>
      <td>35</td>
      <td>4.913514e+06</td>
    </tr>
    <tr>
      <th>235</th>
      <td>riemann</td>
      <td>4.140684e+06</td>
    </tr>
    <tr>
      <th>17</th>
      <td>paradox</td>
      <td>4.119430e+06</td>
    </tr>
    <tr>
      <th>128</th>
      <td>antics</td>
      <td>3.644331e+06</td>
    </tr>
    <tr>
      <th>215</th>
      <td>necessity</td>
      <td>3.605733e+06</td>
    </tr>
    <tr>
      <th>236</th>
      <td>zeta</td>
      <td>3.450111e+06</td>
    </tr>
    <tr>
      <th>255</th>
      <td>314</td>
      <td>3.238687e+06</td>
    </tr>
    <tr>
      <th>257</th>
      <td>de5</td>
      <td>3.238687e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
import matplotlib.pyplot as plt

top_words = avg_views_df.head(20)

plt.figure(figsize=(10,6))
sns.barplot(x='avg_views', y='word', data=top_words)
plt.title("Top 20 Words by Average Views")
plt.xlabel("Average Views")
plt.show()

```


    
![png](README_files/README_18_0.png)
    


We can also see the top 20 videos views and liked and each title


```python
top_views = df.sort_values(by='views', ascending=False).head(20)
top_views[['title', 'views', 'likes']]
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,8))
sns.barplot(x='views', y='title', data=top_views)
plt.title("Top 20 Videos by Views")
plt.xlabel("Views")
plt.ylabel("Title")
plt.show()

```


    
![png](README_files/README_20_0.png)
    


we can see that there are 2 bases of the views ( either the yt vid title is funny /comparative ) or the channel is popular (well known for it's deep analysis)


```python

```
