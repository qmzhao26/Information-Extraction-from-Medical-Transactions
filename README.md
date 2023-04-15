# Information-Extraction-from-Medical-Transactions
Specific information extraction using public medical dataset on kaggle

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


### Reference:
Pre-trained biomedical ner model: https://huggingface.co/d4data/biomedical-ner-all
