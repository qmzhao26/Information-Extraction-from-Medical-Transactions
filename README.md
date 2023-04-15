# Information-Extraction-from-Medical-Transactions
Specific information extraction using public medical dataset on kaggle

### Dataset
Public dataset Medical Transcriptions: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions?resource=download


### Target identification:
* Gender
* Age
* Treatment

Gender: 
  In the description and transcription fields, directly match the possible gender designations
Age: 
  in the description and transcription fields, use regular expressions to match descriptions that contain 'year'
Treatment:
1. traditional NER
2. pre-trained model based on bert's medical specialization to extract NER and match words with entity types A and B
3. multi-label training using transcription and keywords based on the pre-trained model BERT (not yet completed, bugs exist)

### Results
The results are saved in csv file "mtsamples_results.csv":
Resulting files in Google Drive: https://drive.google.com/file/d/1-LKFZH4wLBhZyswkga-xUREWeZ8Lph3t/view?usp=sharing

gender: 'gender' column
age: 'age' column
treatment: 
  1. 'treatment_ngram':  Use the pre-training model 'paraphrase-MiniLM-L6-v2' to get the embedding, and then use the n=gram method to extract the phrases (words) with the highest correlation
  2. 'treatment_wd': Use the textranked method to extract the words with the highest association in "transaction"
  3. 'treatment_trip': SVO triplets extracted manually on the basis of NER obtained from the pre-training model 'en_core_web_sm'
  4. 'treatment_medi': SVO triplets using pre-trained biomedical ner model (based on BERT)
  5. **Note:** BERT-based multi-label classification is not yet complete...


### Reference:
Pre-trained biomedical ner model: https://huggingface.co/d4data/biomedical-ner-all
