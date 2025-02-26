import sys
from dataclasses import dataclass
import os

import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('aritfacts', 'preprocessor.pkl') # giving input to the model
    
class DataTransformation: # This class is used to transform the data
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self): # This function is used for data transformation
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            # creating pipeline for numerical columns that will run on training dataset
            # 1. missing values using median
            # 2. scaling using standard scaler
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False))
            ])
            logging.info("Numerical Columns scaling completed")
            # creating pipeline for categorical columns that will run on training dataset
            # 1. missing values using constant
            # 2. one hot encoding
            categorical_pipeline = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                                                   ('onehot', OneHotEncoder()),
                                                   ('scaler', StandardScaler(with_mean=False))])
            logging.info("Categorical Columns encoding completed")
            
            
            # combining the pipelines using column transformer
            preprocessor = ColumnTransformer(
                [('numerical_pipeline',numerical_pipeline, numerical_columns),
                ('categorical_pipeline',categorical_pipeline, categorical_columns)]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,trian_path,test_path):
        try:
            train_df = pd.read_csv(trian_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("read train and test data")
            logging.info("Obtaining preprocessing object")
            preprocessing_object = self.get_data_transformer_object()
            target_column = 'math_score'
            numerical_columns = ['reading_score', 'writing_score']
            # training dataset
            input_feature_train_df = train_df.drop(target_column, axis=1)
            target_feature_train_df = train_df[target_column]
            # testing dataset
            input_feature_test_df = test_df.drop(target_column, axis=1)
            target_feature_test_df = test_df[target_column]
            logging.info("Fitting the preprocessor on training data")
            
            input_features_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_features_test_arr = preprocessing_object.transform(input_feature_test_df)
            logging.info("Transformation completed")
            
            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Saving the preprocessor object")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_object
            )
            return (train_arr, 
                    test_arr,
                    self.data_transformation_config.preprocessor_ob_file_path)
        except Exception as e:
            raise CustomException(e,sys)
