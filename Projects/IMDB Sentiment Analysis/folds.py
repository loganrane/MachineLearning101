# create folds.py
# import pandas and model_selection module of scikit-learn
import pandas as pd
from sklearn import model_selection

if __name__ == '__main__':
    # Read training data
    df = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
    
    # Map Positive to 1 and Negative to 0
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == 'positive' else 0
    )
    
    # We create a new column called kfold and fill it with -1
    df['kfold'] = -1
    
    # Next we randomize the rows of the data 
    df = df.sample(frac = 1).reset_index(drop = True)
    
    # Fetch Labels
    y = df.sentiment.values
    
    # Initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    # Fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
        
    # Save the new csv with kfold column
    df.to_csv('imdb_folds.csv', index=False)