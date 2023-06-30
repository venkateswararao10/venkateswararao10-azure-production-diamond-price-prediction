import os,sys,numpy as np,pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class datatransformationconfig:
    preprocessorobj_filepath:str=os.path.join('artifacts','preprocessor.pkl')
@dataclass
class datatransformation:
    datatransformconfig=datatransformationconfig
    logging.info('data transformation intiated')
    def getpreprocessorobj(self):
       try: 
        logging.info('getting preprocessor object function intiated')
        # Define which columns should be ordinal-encoded and which should be scaled
        categorical_cols = ['cut', 'color','clarity']
        numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
        # Define the custom ranking for each ordinal variable
        cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
        color_categories = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
        clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
        logging.info('Pipeline Initiated')
        numericalpipeline=Pipeline(
             [
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
             ]
        )
        categoricalpipeline=Pipeline(
            [
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
            ]
        )
        preprocessor=ColumnTransformer(
            [   
                ('numericalpipeline',numericalpipeline,numerical_cols),
                ('categoricalpipeline',categoricalpipeline,categorical_cols)
            ]
        )
        logging.info('preprocessor object function completed')
        logging.info('Pipeline Completed')
        return preprocessor
       except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
    def initiatedatatransformation(self,traindatapath,testdatapath):
       try: 
        traindf=pd.read_csv(traindatapath)
        testdf=pd.read_csv(testdatapath)
        logging.info('Read train and test data completed')
        logging.info(f'Train Dataframe Head : \n{traindf.head().to_string()}')
        logging.info(f'Test Dataframe Head  : \n{testdf.head().to_string()}')
        logging.info('Obtaining preprocessing object')
        preprocessorobj=self.getpreprocessorobj()
        targetcolumn='price'
        dropcolumns=[targetcolumn,'id']
        input_feature_train_df=traindf.drop(dropcolumns,axis=1)
        target_feature_train_df=traindf[targetcolumn]
        input_feature_test_df=testdf.drop(dropcolumns,axis=1)
        target_feature_test_df=testdf[targetcolumn]
        input_feature_train_arr=preprocessorobj.fit_transform(input_feature_train_df)
        input_feature_test_arr=preprocessorobj.transform(input_feature_test_df)
        train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
        test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
        logging.info("Applying preprocessing object on training and testing datasets.")
        save_object(
            obj=preprocessorobj,
            filepath=self.datatransformconfig.preprocessorobj_filepath
        )
        logging.info('pickle file saved')
        return(
            train_arr,test_arr,self.datatransformconfig.preprocessorobj_filepath
        )
       except Exception as e:
           logging.error('exception occured in initiatedatatransform')
           raise CustomException(e,sys)