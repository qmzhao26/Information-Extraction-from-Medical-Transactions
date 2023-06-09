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
2. pre-trained model based on bert's medical specialization to extract NER and match words with entity types 'Therapeutic_procedure' and 'Medician'
3. multi-label training using 'transcription' and 'keywords' columns in the dataset based on the pre-trained model BERT. A simple three-layer network is constructed, with the first layer being the pre-trained BERT (Unlike the direct keyword calculation, the 'keywords' in the dataset contain more medical terms and results. Multi-label classification of them may yield more effective treatment information.)

### Results
Partial results are saved in csv file "mtsamples_results.csv" :
In Google Drive: https://drive.google.com/file/d/1-LKFZH4wLBhZyswkga-xUREWeZ8Lph3t/view?usp=sharing (60.7M)  

**gender**: 'gender' column  
**age**: 'age' column  
**treatment**:   
  1. 'treatment_ngram':  Use the pre-training model 'paraphrase-MiniLM-L6-v2' to get the n-gram embedding (only one word for this proj.), and then use cosine similarity to find the words/phrases that are the most similar to the document
  2. 'treatment_wd': Use the textranked method to extract the words with the highest association in "transaction"
  3. 'treatment_trip': SVO triplets extracted manually on the basis of NER obtained from the pre-training model 'en_core_web_sm'
  4. 'treatment_medi': SVO triplets using pre-trained biomedical ner model (based on BERT[1]) useing label 'Therapeutic_procedure' and 'Medician'
  5. BERT-based multi-label classification result saved in csv file "mtsamples_classify.csv", The 'classification' field stores all predicted labels. **Note:** Due to memory limitations only 128 transcription values are currently used for multi-label prediction, and the results are saved in "mtsamples_classify.csv"  (Only use 500 data trained with 3 epochs)


### Reference:
[1] Pre-trained biomedical ner model: https://huggingface.co/d4data/biomedical-ner-all
