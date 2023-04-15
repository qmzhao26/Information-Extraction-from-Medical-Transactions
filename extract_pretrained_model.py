
import pandas as pd
import time

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

addr = "D:/A Course/interview/archive/mtsamples_added_tradition.csv"
# addr = "D:/A Course/interview/archive/mtsamples_cleaned.csv"

df = pd.read_csv(addr)

# using pre-trained biomedical NER model and tokenizer
# model link: https://huggingface.co/d4data/biomedical-ner-all
tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")
pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0) #using gpu

# Add as treatment If the entity category is 'Therapeutic_procedure' or 'Medication'
def get_medi(text):
    result = pipe(text)
    medi = []
    for i in result:
        if i['entity_group'] == 'Therapeutic_procedure' or i['entity_group'] == 'Medication':
            medi.append(i['word'])
    return medi

# Apply all transcription data to pre-trained biomedical-ner model
def add_medi():
    df['treatment_medi'] = df['transcription'].apply(lambda x : get_medi(x))
    return df

t1 = time.time()
df = add_medi()
print('time:',time.time()-t1)

df.to_csv("D:/A Course/interview/archive/mtsamples_added_medi.csv", index=False)

df.info()

# print(df.at[4, 'transcription'])
# print("Triplets:")
# print(df.at[4, 'treatment_trip'])
# print("N-gram:")
# print(df.at[4, 'treatment_ngram'])
# print("Medi:")
# print(df.at[4, 'treatment_medi'])


