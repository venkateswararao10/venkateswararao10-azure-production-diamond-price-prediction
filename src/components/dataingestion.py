import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
@dataclass
class dataingestionconfig:
    traindatapath:str=os.path.join('artifacts','train.csv')
    testdatapath:str=os.path.join('artifacts','test.csv')
    rawdatapath:str=os.path.join('artifacts','raw.csv')
@dataclass
class dataingestion:
    ingestionconfig=dataingestionconfig
    def initiatedataingestion(self)->(str,str):
        logging.info('data ingestion starts')
        try:
            df=pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            logging.info('gemstone dataset read as pandas dataframe')
            os.makedirs(os.path.dirname(self.ingestionconfig.rawdatapath),exist_ok=True)
            df.to_csv(self.ingestionconfig.rawdatapath,index=False)
            traindata,testdata=train_test_split(df,test_size=0.30,random_state=42)
            logging.info('train_test_split is done')
            traindata.to_csv(self.ingestionconfig.traindatapath,index=False)
            testdata.to_csv(self.ingestionconfig.testdatapath,index=False)
            logging.info('data ingestion completed')
            return(
                self.ingestionconfig.traindatapath,
                self.ingestionconfig.testdatapath
            )
        except Exception as e:
            logging.error('error occured in data ingestion')
            raise CustomException(e,sys)