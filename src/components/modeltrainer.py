import os,sys,numpy as np,pandas as pd
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model
from dataclasses import dataclass
@dataclass
class modeltrainerconfig:
    modeltrainer_filepath=os.path.join('artifacts','model.pkl')
@dataclass
class modeltraining:
    modeltrainerpath=modeltrainerconfig
    def initiatemodeltrainer(self,train_arr,test_arr):
      try:
        logging.info('model trainer initiated')
        logging.info('Splitting Dependent and Independent variables from train and test data')
        X_train,X_test,y_train,y_test=(
            train_arr[:,:-1],test_arr[:,:-1],train_arr[:,-1],test_arr[:,-1]
           )
        models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
            }
        report,modelreport=evaluate_model(X_train, X_test, y_train, y_test, models)
        logging.info(f'Model Report : {report}')
        bestr2score=max(sorted(list(report.values())))
        bestmodelname=list(models.keys())[list(report.values()).index(bestr2score)]
        bestmodel=models[bestmodelname]
        bestmodel1=modelreport[bestmodelname]
        bestmodel2=[bestmodel,bestmodel1]
        logging.info(f'best model 2 {bestmodel2}')
        logging.info(f'Best Model Found , Model Name : {bestmodelname} , R2 Score : {bestr2score}')
        save_object(obj=bestmodel2, filepath=self.modeltrainerpath.modeltrainer_filepath)
        logging.info('modeltrainercompleted')
      except Exception as e:
        logging.error('error occured in model trainer')
        raise CustomException(e, sys)