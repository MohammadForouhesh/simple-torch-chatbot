"""
1- Downloading the Microsoft WikiQA corpus:
The Microsoft WikiQA corpus is available for download on the Microsoft Research website. We downloaded the corpus from the /
following link: https://www.microsoft.com/en-us/download/details.aspx?id=52419.

2- Preprocessing the Microsoft WikiQA corpus:
Once you have downloaded the corpus, we preprocess it to make it ready for training your chatbot model. /
Here are some common preprocessing steps that we performed:

    * Remove any unnecessary metadata or formatting from the corpus
    * Split the corpus into training and validation sets
    * Tokenize the text into individual words or subwords
    * Convert the text to numerical values using word embeddings
    * Create batches of data to feed into the model during training and validation steps 
"""