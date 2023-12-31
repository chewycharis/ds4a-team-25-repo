# Proof-of-Concept Application: <br> An Intelligent Agent for Job Postings Recommendation Based on Resume 

This is a DS4A/Women project by Team 25 created from June to July of 2023. 
<br>For more about the program, please visit https://www.correlation-one.com/data-skills-for-all.
<br>For visualizations on our database, please check out this [Tableau dasboard](https://public.tableau.com/app/profile/chuhui.fu/viz/Book1_16895323042590/Dashboard1?publish=yes). 
<br>For a one-pager on the entire project, please take a look at the [datafolio](https://github.com/chewycharis/ds4a-team-25-repo/blob/main/Datafolio.pdf). For more details, check out [these google slides](https://docs.google.com/presentation/d/1G7E3fRn6jOg0URo1g-cW5dPgTBEamaQ_ng7XOijWZLA/edit?usp=sharing).

### How to use this app? 
<ol>
<li>Select a type of data jobs.</li>
<li>Submit your resume in PDF format. </li>
<li>Obtain top job postings. </li>
<li>[Optional] Obtain a summary of the job description. </li>
<li>[Optional] Obtain resume improvement suggestions.</li>
</ol>

### Example Outputs: 
![summary](https://github.com/chewycharis/ds4a-team-25-repo/blob/main/images/summary.png?raw=true)
![improve resume](https://github.com/chewycharis/ds4a-team-25-repo/blob/main/images/improve.png?raw=true)

### How did we build this app? 
We cleaned up data scraped by [picklesueat](https://github.com/picklesueat/data_jobs_data) in June 2020. Following that, we used the sentence transformer model ("all-MiniLM-L6-v2") via BerTopic package to create embeddings for job descriptions, and used cosine similarity to find job descriptins most similar to resumes submitted by the user. We also allowed for the option to call the OpenAI API to create responses using gpt-3.5.


