from src.components.dataingestion import dataingestion
from src.components.datatransformation import datatransformation
from src.components.modeltrainer import modeltraining
if  __name__=='__main__':
 dataingestion=dataingestion()
 traindatapath,testdatapath=dataingestion.initiatedataingestion()
 datatransform=datatransformation()
 train_arr,test_arr,preprocessorobj_filepath=datatransform.initiatedatatransformation(traindatapath, testdatapath)
 modeltrainer=modeltraining()
 modeltrainer.initiatemodeltrainer(train_arr, test_arr)