import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('aritfacts', 'train.csv')
    test_data_path: str=os.path.join('aritfacts', 'test.csv')
    raw_data_path: str=os.path.join('aritfacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # reading the raw data from this line below (can be changed api,ui,database,atlas...)
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Data read successfully as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # Create the directory if it does not exist
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True) # Save the raw data
            logging.info("Raw data saved successfully")
            
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True) # Save the train data
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) # Save the test data
            logging.info("Data Ingestion completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
   
   
   
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion() 
    data_transformation = DataTransformation()
    train_arr, test_arr, pp_obj_path = data_transformation.initiate_data_transformation(train_data,test_data)
    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
           