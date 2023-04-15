
import pandas as pd
import re
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords

addr = "D:/A Course/interview/archive/mtsamples.csv"

df = pd.read_csv(addr)

# view = df.isnull().sum().sort_values(ascending = False)

df = df[df['transcription'].notna()]
df.info()

# Word Count in Medical Specialty
ms_list = []
wc_list =[]
for specialty in df['medical_specialty'].unique():
    df_filter = df.loc[(df['medical_specialty'] == specialty)]
    wc_tmp = df_filter['transcription'].str.split().str.len().sum()
    ms_list.append(specialty)
    wc_list.append(wc_tmp)
wc_df = pd.DataFrame({'Medical_Specialty':ms_list, 'Word_Count':wc_list})
wc_df['Word_Count'] = wc_df['Word_Count'].astype('int')
wc_df = wc_df.sort_values('Word_Count', ascending=False)
wc_df.reset_index(drop=True)
print(wc_df)

# Lowercase all words
def lower(df, attribute):
    df['new'+"_"+attribute] = df[attribute].apply(lambda x : str.lower(x))
    return df

df = lower(df,'transcription')
df = lower(df,'description')

# Remove all Punctuation
def remove_punc_num(df, attribute):
    df.loc[:,attribute] = df[attribute].apply(lambda x : " ".join(re.findall('[\w]+',x)))
    df[attribute] = df[attribute].str.replace('\d+', '')
    return df
df = remove_punc_num(df, 'new_transcription')
df = remove_punc_num(df, 'new_description')

# Tokenise
tk = WhitespaceTokenizer()
def tokenise(df, attribute):
    df[attribute] = df.apply(lambda row: tk.tokenize(str(row[attribute])), axis=1)
    return df
df = tokenise(df, 'new_transcription')
df = tokenise(df, 'new_description')

# Stemming
from nltk.stem.snowball import SnowballStemmer
def stemming(df, attribute):
    stemmer = SnowballStemmer("english")
    df[attribute] = df[attribute].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word.
    return df
df = stemming(df, 'new_transcription')
df = stemming(df, 'new_description')

# Remove Stop Words
def remove_stop_words(df, attribute):
    stop = stopwords.words('english')
    df[attribute] = df[attribute].apply(lambda x: ' '.join([word for word in x if word not in (stop)]))
    return df
df = remove_stop_words(df, 'new_transcription')
df = remove_stop_words(df, 'new_description')

df.info()

# Save to csv
df.to_csv('D:/A Course/interview/archive/mtsamples_cleaned.csv', index=False)