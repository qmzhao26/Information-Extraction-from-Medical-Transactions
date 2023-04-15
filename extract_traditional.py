
import pandas as pd
import numpy as np
import re
import time
import os

from nltk.tokenize import WhitespaceTokenizer

import spacy
import pytextrank
from keybert import KeyBERT
import textacy.extract


addr = "D:/A Course/interview/archive/mtsamples_cleaned.csv"

df = pd.read_csv(addr)

# df = df[0:1000]

# Here is the gender word dictionary
female = {"her", "she", "herself", "hers", "woman", "female", "girl", 'lady', 'miss'}
male = {'guy','dude','gentleman','male','boy','he', 'his', 'men'}

# Load the pre-trained spaCy model for English
# Used for NER analysis
nlp = spacy.load('en_core_web_sm')

# Load pre-trained BERT model
model = KeyBERT('paraphrase-MiniLM-L6-v2')

# Only proceed to tokenize, do not do evaporation and stop word deletion
# For gender and age matching
def clean(df, attribute):
    tk = WhitespaceTokenizer()
    df['clean'+"_"+attribute] = df[attribute].apply(lambda x : str.lower(x))
    df.loc[:,'clean'+"_"+attribute] = df['clean'+"_"+attribute].apply(lambda x : " ".join(re.findall("^['\"{}\\(\\)\\[\\]\\*&.?!,…:;]+$",x)))
    # df.loc[:,'clean'+"_"+attribute] = df['clean'+"_"+attribute].apply(lambda x : nlp.tokenizer(x))
    df['clean'+"_"+attribute] = df['clean'+"_"+attribute].str.replace('\d+', '')
    df['clean'+"_"+attribute] = df.apply(lambda row: tk.tokenize(str(row[attribute])), axis=1)
    return df

df = clean(df,'transcription')
df = clean(df,'description')

# Matching Gender
def add_gender():
    df['gender_from_trans'] = df['clean_transcription'].apply(lambda x : 1 if len(set(x) & male)!=0 else (2 if len(set(x) & female)!=0 else 3))
    df['gender_from_desc'] = df['clean_description'].apply(lambda x : 1 if len(set(x) & male)!=0 else (2 if len(set(x) & female)!=0 else 3))
    # Merge Results
    df['gender'] = df['gender_from_trans']
    df['gender'] = np.where(df['gender']==3,df['gender_from_desc'],df['gender'])
    df['gender'] = df['gender'].apply(lambda x : 'male' if x==1 else ("female" if x==2 else 'unknown'))
    return df
df = add_gender()
print(df[['gender']].head(10))

# Match the age that contains 'year'
def add_age():
    df['age_trans'] = df['transcription'].apply(lambda x : re.findall("\d+-?year",x))
    df['age_desc'] = df['description'].apply(lambda x : re.findall("\d+-?year",x))
    # Merge Results
    df['age'] = df['age_trans']
    df['age'] = np.where(df['age'],df['age'],df['age_desc'])
    return df
df = add_age()
print(df[['age']].head(10))



# Show full dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',150)



# Manual triplet extraction formula
def extract_triplets(doc):
    subject = ""
    relation = ""
    object = ""
    compound = ""
    # beforeroot = ""
    # afterroot = ""
    print('\n')
    tri_list = []
    subject_set = set()
    for token in doc:
        print(token, token.dep_)
        if token.dep_ == 'nsubj' and subject != "":
            if relation == "":
                subject_set.add(subject)
                subject_set.add(object)
            else:
                tri_list.append((subject, relation, object))
            subject = token.text
            relation = ""
            object = ""
            compound = ""
        else:
            if token.dep_ == 'nsubj':
                subject = compound+token.text
                compound = ""
            elif token.dep_ == 'ROOT':
                relation = token.text
                compound = ""
            elif token.dep_ == 'dobj':
                object = compound+token.text
                compound = ""
            elif token.dep_ == 'compound':
                compound += token.text+' '
            else:
                compound = ""
    if relation == "":
        subject_set.add(subject)
        subject_set.add(object)
    elif subject != "":
        tri_list.append((subject, relation, object))
    return tri_list, subject_set

# Extract triplets from text
def get_triplets(text):
    doc = nlp(text)
    triplets = []
    allsubs = set()
    for sentence in doc.sents:
        tri, subset = extract_triplets(sentence)
        allsubs = allsubs.union(subset)
        if tri:
            triplets.append(tri)
    # stemmer = SnowballStemmer("english")
    # stemed_sub = [stemmer.stem(y) for y in allsubs]
    if len(allsubs) != 0:
        triplets.append(allsubs)
    return triplets

# Extract triplets using library 'textacy'
def new_get_triplets(text):
    doc = nlp(text)
    tris = []
    svo_triples = textacy.extract.triples.subject_verb_object_triples(doc)
    for triple in svo_triples:
        tris.append((triple[0], triple[1], triple[2]))
    return tris

# Extract top-ranked words
def get_words(text):
    doc = nlp(text)
    words = []
    for phrase in doc._.phrases:
        words.append(phrase.chunks[0])
    return words

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Extract top-ranked words using pre-trained BERT in library 'keybert'
def get_ngram_bert(text):
    keywords = model.extract_keywords(text, keyphrase_ngram_range=(1,1),  nr_candidates=20, top_n=10)
    return keywords

# Add triplets and top-ranked words to dataframe using multiple methods
def add_treatment():
    df['treatment_trip'] = df['transcription'].apply(lambda x : new_get_triplets(x))
    df['treatment_ngram'] = df['transcription'].apply(lambda x : get_ngram_bert(x))
    nlp.add_pipe("textrank")
    df['treatment_wd'] = df['transcription'].apply(lambda x : get_words(x))
    return df


# Add treatment
t1 = time.time()
df = add_treatment()
print("time for treatment extraction: ", time.time()-t1)

# save to csv
df.to_csv("D:/A Course/interview/archive/mtsamples_added_tradition.csv", index=False)

