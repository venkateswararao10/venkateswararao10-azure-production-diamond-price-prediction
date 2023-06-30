import os,sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
from src.utils import load_object


class customdata:
    def __init__(self, carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
             self.carat=carat
             self.cut=cut
             self.color=color
             self.clarity=clarity
             self.depth=depth
             self.table=table
             self.x=x
             self.y=y
             self.z=z
    def get_data_as_dataframe(self):
        try:
            customdata_inputdict ={
                'carat':[self.carat],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z]}
            df=pd.DataFrame(customdata_inputdict)
            logging.info('dataframe is created')
            return df
        except Exception as e:
            logging.error('error occured in custom data function')
            raise CustomException(e, sys)
@dataclass
class predictpipeline:
    def predict(self,features):
      try:  
        preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
        model_path=os.path.join('artifacts','model.pkl')
        preprocessor=load_object(preprocessor_path)
        model=load_object(model_path)
        data_scaled=preprocessor.transform(features)
        #pred=model[1].predict(data_scaled) or
        pred=model[0].predict(data_scaled)
        logging.info('predictpipeline is completed')
        return pred
        
      except Exception as e:
        logging.error('error occured in predict function')  
        raise CustomException(e, sys)

if __name__=='__main__':
    
     cd=customdata(carat=1.52,
            depth = 58.0 ,
            table = 67,
            x = 3,
            y = 4,
            z = 5,
            cut ='Premium' ,
            color= 'F',
            clarity ='VS2')
     df=cd.get_data_as_dataframe()
     obj=predictpipeline()
     result=obj.predict(df)
     print(float(result))