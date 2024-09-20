import pandas as pd
#from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import RandomOverSampler

class DataLoader():
    def __init__(self):
        self.train_data = None

    def load_dataset(self):
        self.train_data = pd.read_csv("data/titanic-full.csv")

    def preprocess_data(self):
        self.train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
        self.train_data['Age'] = self.train_data['Age'].fillna(self.train_data['Age'].median())
        categorical_cols = ["Sex", "Embarked"]
        self.train_encoded = pd.get_dummies(self.train_data[categorical_cols], prefix=categorical_cols)
        self.train_data = pd.concat([self.train_encoded, self.train_data], axis=1)
        self.train_data.drop(categorical_cols, axis=1, inplace=True)
        
        # Explicitly cast the new 'Embarked' columns to int64
        self.train_data['Embarked_S'] = self.train_data['Embarked_S'].astype('int64', errors='ignore')
        self.train_data['Embarked_C'] = self.train_data['Embarked_C'].astype('int64', errors='ignore')
        self.train_data['Embarked_Q'] = self.train_data['Embarked_Q'].astype('int64', errors='ignore')
        

    def get_data_split(self):
        X = self.train_data.iloc[:, :-1]
        y = self.train_data.iloc[:, -1]
        return X, y
