# create train.py
import io
import torch

import numpy as np
import pandas as pd

# We will use TensorFlow but not for training
import tensorflow as tf

from sklearn import metrics

# The other modules to import when using via console
# import config
# import dataset
# import engine
# import lstm

def load_vectors(fname):
    # taken from: https://fasttext.cc/docs/en/english-vectors.html
    fin = io.open(
        fname,
        'r',
        encoding='utf-8',
        newline='\n',
        errors='ignore'
    )
    
    n, d = map(int, fin.readline().split())
    
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    
    return data

def create_embedding_matrix(word_index, embedding_dict):
    """
    This function created the embedding matrix.
    :param word_index: a dictionary with word: index_value
    :param embedding_dict: a dictionary with word: embedding vector
    :return: a numpy array with embedding vectors for all known words
    """
    
    # Initialize the matrix with zeroes
    embedding_matrix = np.zeros((len(word_index + 1), 300))
    
    # loop over all the words
    for word, i in word_index.items():
        # if word is found in pre-trained embeddings, 
        # update the matrix. if the word is not found, 
        # the vector is zeros
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]
        
        # return the embedding matrix
        return embedding_matrix
    

def run(df, fold):
    """
    Run training and validation for a given fold and dataset
    :param df; pandas dataframe with kfold column
    :param fold: current fold, int
    """
    
    # fetch training dataframe
    train_df = df[df.kfold != fold].reset_index(drop = True)
    
    #fetch validation dataframe
    valid_df = df[df.kfold == fold].reset_index(drop = True)
    
    print('Fitting Tokenizer')
    # We use tf.keras for tokenization
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())
    
    # Convert training data to sequences
    # for example : "bad movie" gets converted to
    # [24, 27] where 24 is the index for bad and 27 is the
    # index for movie
    xtrain = tokenizer.texts_to_sequences(train_df.review.values)
    
    # convert validation data to sequences
    xtest = tokenizer.texts_to_sequences(valid_df.review.values)
    
    # zero pad the training sequences given the maximum length this padding is done on left
    # hand side
    # if sequence is > MAX_LEN, it is trucated on left hand side too
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(
#         xtrain, maxlen=config.MAX_LEN
        xtrain, maxlen=MAX_LEN
    )
    
    # zero pad the validation sequences
    xtest = tf.keras.preprocessing.sequence.pad_sequences(
#         xtest, maxlen = config.MAX_LEN
        xtest, maxlen = MAX_LEN
    )
    
    # initialize dataset class for training
#     train_dataset = dataset.IMDBDataset(
    train_dataset = IMDBDataset(
        reviews = xtrain, 
        targets = train_df.sentiment.values
    )
    
    # Create torch dataloader for training
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
#         batch_size = config.TRAIN_BATCH_SIZE,
        batch_size = TRAIN_BATCH_SIZE,
        num_workers = 2
    )
    
    # Initialize dataset class for validation
#     valid_dataset = dataset.IMDBDataset(
    valid_dataset = IMDBDataset(
        reviews = xtest, 
        targets = valid_df.sentiment.values
    )
    
    # Create torch dataloader for training
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
#         batch_size = config.VALID_BATCH_SIZE,
        batch_size = VALID_BATCH_SIZE,
        num_workers = 1
    )
    
    print('Loading Embeddings')
    # Load embeddings as shown previously
    embedding_dict = load_vectors('crawl-300d-2M.vec')
    embedding_matrix = create_embedding_matrix(
        tokenizer.word_index, embedding_dict
    )
    
    # Create torch device, since we are using GPU, we will use cuda
    device = torch.device('cuda')
    
    # Fetch out LSTM model
#     model = lstm.LSTM(embedding_matrix)
    model = LSTM(embedding_matrix)
    
    # Send model to device
    model.to(device)
    
    # initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print('Training Model')
    # Set best accuracy to 0
    best_accuracy = 0
    # Set early stopping counter to zero
    early_stopping_counter = 0
    # train and validation for all epochs
#     for epoch in range(config.EPOCHS):
    for epoch in range(EPOCHS):
        #train one epoch
        engine.train(train_data_loader, model, optimizer, device)
        # validate
        outputs, targets = engine.evaluate(
                                    valid_data_loader, model, device
                                )
        
        # use threshold of 0.5
        # we are using linear functions and not sigmoid
        outputs = np.array(outputs) >= 0.5
        
        # Calculate accuracy
        accuracy = metrics.accuracy_score(targets, outputs)
        print(
             f"FOLD:{fold}, Epoch: {epoch}, Accuracy Score = {accuracy}"
        )
        
        # simple early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            early_stopping_counter += 1
        if early_stopping_counter > 2:
             break
                
if __name__ == '__main__':
    
    # load data
    df = pd.read_csv('imdb_folds.csv')
    
    # train for all folds
    run(df, fold=0)
    run(df, fold=1) 
    run(df, fold=2)  
    run(df, fold=3)
    run(df, fold=4)