import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import pandas as pd 
import os
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")



def suggestion_llm(clean_resume, job_description):
  content = f"""
  Please keep your response to 256 tokens without cut-off sentences.
  Here is my resume: \n
  {clean_resume} \n
  Give me suggestions on how I can improve my resume for this job description: \n
  {job_description}
  """

  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages= [{"role":"user", "content" : content}],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  return response 

def summary_llm(dirty_job_description):
  content = f"""
  Please keep your response to 256 tokens without cut-off sentences.
  Give me a brief summary of job description, be sure to highlight the responsibilities, required skills and required experiences: \n
  {dirty_job_description}
  """
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages= [{"role":"user", "content" : content}],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  return response 

def determine_role(user_input):
    if "data analyst" in user_input:
        return "data analyst"
    if "data engineer" in user_input:
        return "data engineer"
    if "data scientist" in user_input:
        return "data scientist"


def present_jobs(top_5_jobs_df, k=0):
    top_job = top_5_jobs_df.iloc[k]
    print("Company Name:", top_job["Company Name"])
    print("Job Title:", top_job["Job Title"])
    print("Location:", top_job["Location"])
    print("Industry:", top_job["Industry"])
    print("Company Size:", top_job["Size"])
    print("Salary Estimate ($):", top_job["Salary_Estimate_Lower_Bound"] ,"-", top_job["Salary_Estimate_Upper_Bound"])
    print("Original Job Description:\n", top_job["Job Description"])

def find_top_k_jobs(clean_resume, role= "data analyst", k=5):
   if role == "data analyst":
      job_postings = pd.read_csv("./data/data_analyst_job_postings_local_dirty.csv")
      job_postings_embeddings = np.load("./data/data_analyst_job_postings_embeddings.npy")
   if role =="data engineer":
      job_postings = pd.read_csv("./data/data_engineer_job_postings_local_dirty.csv")
      job_postings_embeddings = np.load("./data/data_engineer_job_postings_embeddings.npy")
   if role == "data scientist": 
      job_postings = pd.read_csv("./data/data_scientist_job_postings_local_dirty.csv")
      job_postings_embeddings = np.load("./data/data_scientist_job_postings_embeddings.npy")

   sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
   resume_embeddings = sentence_model.encode([clean_resume], show_progress_bar=False)
   cs = cosine_similarity(job_postings_embeddings,resume_embeddings, dense_output=True)
   top_job_post_index = cs.flatten().argsort()[::-1][:k]
   return job_postings.iloc[top_job_post_index]


def clean_text(text_string,lemmatize=False):
  text_data = re.sub('[^a-zA-Z]', ' ', text_string)
  text_data = text_data.lower()
  text_data = text_data.split()
  clean_text = ' '.join(text_data)
  if lemmatize:
    wl = WordNetLemmatizer()
    text_data = [wl.lemmatize(word) for word in text_data if not word in set(stopwords.words('english'))]
    text_data = ' '.join(text_data)
    return text_data
  else:
    return clean_text
  
def get_resume_str(path):
    pdf = PdfReader(path)
    page = pdf.pages[0]
    text = page.extract_text()
    return text 
  
def resume_cleaning(path):
    raw_resume = get_resume_str(path)
    return clean_text(raw_resume, lemmatize=True)
