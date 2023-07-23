import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np 
import pandas as pd 

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

from helper import resume_cleaning, find_top_k_jobs, present_jobs, determine_role, suggestion_llm, summary_llm


def main():
    user_input_role = input("What role are you looking for? \nOur database currently have job postings for data analyst, data engineer, and data scientists roles. Please pick one.\n")
    
    if not any(r in user_input_role.lower() for r in ["data analyst", "data engineer", "data scientist"]):
        print(f"Sorry, but our job postings database currently does not have postings this role.")
        exit()

    role = determine_role(user_input_role)
    print(f"It's great that you are taking steps towards your next role!\nLet me help you find a {role} role that best fits you.")
    raw_resume = input("To find a role that fits you, please paste a directory to your pdf resume under here:\n")
    print("Resume text cleaning in progress ...")
    try:
        clean_resume = resume_cleaning(raw_resume)
        print("Done.")
    except:
        print("We are having trouble parsing your resume at this time. Please ensure that your resume does not contain image or shapes.")
        exit()
    print("Matching resume to job postings in our database ...")
    try:
        top_5_jobs_df = find_top_k_jobs(clean_resume, role)
        print("Done.")
    except:
        print("We are having trouble matching your resume to job postings. Come back another time.")
        exit()
    
    print("Here is the first job posting we found in our data base that matches your resume the best:")
    k=0
    present_jobs(top_5_jobs_df,k)

    user_input_summary = input("Would you like a summarized version of this job description? (Yes/No)\n")
    if user_input_summary.lower() == "yes":
        print("Summary in progress...")
        dirty_job_description = top_5_jobs_df.iloc[k]["Job Description"]
        summary_response = summary_llm(dirty_job_description)
        try:
            summary = summary_response.choices[0].message.content
            print(summary)
        except:
            print ("Sorry, something went wrong.")
        print("\n")

    user_input_suggestion = input("Would you like suggestions on improving your resume for this job? (Yes/No) \n")
    if user_input_suggestion.lower() == "yes":
        print("Suggestion generation in progress...")
        job_description = top_5_jobs_df.iloc[k]["lemmatized_job_description"]
        suggestion_response = suggestion_llm(clean_resume, job_description)

        try:
            suggestion = suggestion_response.choices[0].message.content
            print(suggestion)
        except:
            print ("Sorry, something went wrong.")
        print("\n")
    k+=1
    while k<5:
        user_input_another_job = input("Would you like to see another job? (Yes/No)\n")
        if user_input_another_job.lower() == "no":
            print("Okay. Thank you so much for using me! Have a good day :) \n")
            exit()
        else:
            print("Here is the another job posting we found in our data base that matches your resume well:")
            present_jobs(top_5_jobs_df,k)
            user_input_summary = input("Would you like a summarized version of this job description? (Yes/No)\n")
            if user_input_summary.lower() == "yes":
                print("Summary in progress...")
                dirty_job_description = top_5_jobs_df.iloc[k]["Job Description"]
                summary_response = summary_llm(dirty_job_description)
                try:
                    summary = summary_response.choices[0].message.content
                    print(summary)
                except:
                    print ("Sorry, something went wrong.")
                print("\n")

            user_input_suggestion = input("Would you like suggestions on tailoring your resume to this job? (Yes/No) \n")
            if user_input_suggestion.lower() == "yes":
                print("Suggestion generation in progress...")
                job_description = top_5_jobs_df.iloc[k]["lemmatized_job_description"]
                suggestion_response = suggestion_llm(clean_resume, job_description)
                try:
                    suggestion = suggestion_response.choices[0].message.content
                    print(suggestion)
                except:
                    print ("Sorry, something went wrong.")
                print("\n")
            k+=1
    print("That was the last of the top 5 jobs!")
    print("Thank you so much for using me! Have a good day :) \n")
    


if __name__ == "__main__":
    main()