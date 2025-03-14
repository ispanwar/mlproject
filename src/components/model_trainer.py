import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('aritfacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting the training and test input data")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours": KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "Adaboost Regressor": AdaBoostRegressor()            
            }
            #! add parameters for hyper parameter tuning and refer repo
            #********************************************#
            # getting report of all the models
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test = X_test, y_test = y_test, models = models)
            logging.info("Got the report of all model")
            # to get the best model score from dict
            best_model_score = max(sorted(model_report.values()))
            logging.info("Got the best model score")
            # to get the name of top scorer
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info("GOt the top performer")
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best model found")
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            logging.info("best model pkl created")
            
            predicted = best_model.predict(X_test)
            r2score = r2_score(y_test,predicted)
            return r2score
        except Exception as e:
            raise CustomException(e,sys)