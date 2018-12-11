## Topic: Do Popular Songs Endure?                 

#### Problem Statement 
As a fan of 1960's popular music, I have noticed (I think) that a surprising number of the most enduring songs (as measured by current sales or airplay) were not at the top of the charts when then came out.    By analysis of pop charts and other data, can you measure to what extent is this observation is true?

The key here is that first we need an accurate model to predict the current popularity of a recording which appeared at position p on the charts w weeks ago?   Clearly this increases with p, and decreases (likely exponentially) with w.   Your first job is to gather data (ideally going all the way back to the 1950’s to today) on the popularity of songs at their time of release (like Billboard rankings or sales), and then measures of contemporary popularity (perhaps Spotify downloads, Youtube views, etc.) to build such a model and measure its accuracy over the entire time interval.    Be careful to do cleaning to make sure that you link the right recording of a song to its Billboard rank -- many songs get recorded by multiple groups, and many recordings appear repeated on different albums after this release.

After this, you have a tool which can quantify for each recording how much more/less popular it is than it “should be”.     You must learn enough about popular music of different time periods (each decade from the 1950’s through the present day to be able to perform a sniff test of whether you are capturing real phenomena.  Read some histories, and identify what the major groups from each period are.   Listen to the songs from each interval which your methods identify as outliers in current popularity, and see if you can understand why.  You should learn enough to identify major groups/artists and form ideas about them -- the Beatles, Grand Funk, the Beach Boys, the Bee Gees, Kiss, Sonny and Cher, Michael Jackson, Britney Spears. Last year’s teams did a miserable job on this project because they refused to do sniff tests of their computational results.   As an example of songs whose fate has diverged with time, the number 1 Billboard song of 1966 was “The Ballad of the Green Berets”, while number 36 was “Good Vibrations”.

After you have an effective model to identify over and underperforming songs, there are a variety of issues to explore to try to explain why, including:

For each time period report the 10 songs with the dramatically highest/lowest performance.  Look for properties that explain what you see -- make hypothesis, do statistical tests, and draw conclusions.
For popular groups with many songs on the charts, what is the trend over their career.   Do their early/late songs tend to over/under perform?    Does a popular group’s lesser hits retain more/less popularity than their top hits?
How is over/under-performance a function of the number of charted songs they have?  Do one-hit wonders retain more/less popularity than expected by their chart positions?   Does this differ depending upon how high their charted songs were?   Realize that there is a power law involved here: the #1 song is more popular than #2 by a much bigger margin than the #63 song is over #64.
Consider the effects of genre: how have different genres performed?   Do movie themes or dance songs behave differently.
Consider other data sources, like song lyrics and the Million song data.    Can you make better explanations of why songs endured or failed?

There is room for similar studies of other forms of entertainment media: best selling books, movies, Bollywood music -- but I need you to develop your models first for the music before moving on.


#### Approach 
1. Get the Billboard Top 100 Songs using the Billboard API for the years from 1958 to 2018.
2. Then using the Spotify API, we got the Spotify Popularity for each song on the Billboard for Rank 1 to 100 songs for each week
3. We had two approaches after this: 

   i) For the Midway Report: 
   > After we have gathered the data from billboard into the CSV file, we grouped the data by
different titles and added a new column for the year so that we can later extract the data
based on year. Then we extracted the first
appearance and last appearance of the song in the Billboard chart. Then using the appearances we found the years for which the song actually endured and picked the songs that are endured for a
long time. We then found a list of pairs the song appeared and it's name and sorted it. Now combining, we picked the songs in such a way that the songs are endured for the
longest time and have appeared most of the time in the dataset to make the power law curve for
the baseline model as accurate as possible. We were able to successfully identify 5 recordings in 1960 that have endurance in 2018
![Power Law](https://github.com/jaytorasakar8/Data-Science-Fundamentals/blob/master/Final%20Project/Plots%20-%20Midway/Power%20Law%201.png)

    ii) For the Final Report:
    > Here we were using the Billboard data set and the song's popularity for all years. Now for each song we found the year of release and plotted it against the current Spotify Popularity, and then we found the outliers. The outliers are basically the songs which are not performing on the same lines as other songs. 
    We analyzed the songs to why a song was underperforming or overperforming using a Linear Regression Model.
    Then we were using the Youtube API to get the insights into why the songs were under performing or over performing as compared to others. We also used other API and websites like Discogs API, MusicBrainz API and wikipedia pages to get more data about the songs.
    ![Final Report](https://github.com/jaytorasakar8/Data-Science-Fundamentals/blob/master/Final%20Project/Plots%20-%20Final%20Report/Comparison%20Rank%201%2C2%2C50%2C75%2C99%20Songs.png)
