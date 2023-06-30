import os,sys
import pickle
import numpy as np 
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
def save_object(obj:str,filepath:str):
            try:
                os.makedirs(os.path.dirname(filepath),exist_ok=True)
                f=open(filepath,'wb')
                pickle.dump(obj, f)
                f.close()
                logging.info('succesfully dump the preprocessorobj')
            except Exception as e:
                logging.error('error occured in while dumping preprocessorobj in pickle file')
                raise CustomException(e,sys)

def evaluate_model(X_train,X_test,y_train,y_test,models):
  try:  
    report=dict()
    modelreport=dict()
    for i in range(len(models)):
        model=list(models.values())[i]
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        r2=r2_score(y_test, y_pred)
        report[list(models.keys())[i]]=r2
        modelreport[list(models.keys())[i]]=model 
        logging.info('evalute model is completed') 
    return report,modelreport
    
  except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)

def load_object(filepath):
  try:
      f=open(filepath,'rb')
      r=pickle.load(f)
      f.close()
      logging.info('load_object is completed')
      return r
  except Exception as e:
    logging.error('error occured in load_object')
    raise CustomException(e, sys)
  

