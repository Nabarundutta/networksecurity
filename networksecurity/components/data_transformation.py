import sys 
import os 
import numpy 
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import pickle
import numpy as np

from networksecurity.constants.training_pipeline import TARGET_COLUMN
from networksecurity.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from networksecurity.entity.artifact_entity import(
    DataTransformationArtifact,
    DataValidationArtifact
)
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data,save_object

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
        try:
            self.data_validtation_artifact:DataValidationArtifact = data_validation_artifact
            self.data_transformation_config:DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    ## created function for reading the testing and training data 
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def get_data_transfomer_object(cls)->Pipeline:
        logging.info("Entered get_data_transformer_object method of transformation class")
        try:
            imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info("Initialised knn imputer ")
            processor:Pipeline = Pipeline([("imputer",imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            train_df = DataTransformation.read_data(self.data_validtation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validtation_artifact.valid_test_file_path)
            ### training dataframe 
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            input_feature_train_df=input_feature_train_df.replace(-1,0)
            target_feature_train_df = target_feature_train_df.replace(-1,0)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            input_feature_test_df=input_feature_test_df.replace(-1,0)
            target_feature_test_df = target_feature_test_df.replace(-1,0)

            ##implement knn imputer
            processor = self.get_data_transfomer_object()
            preprocessor_obj = processor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_obj.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_obj.transform(input_feature_test_df)

            train_array = np.c_[transformed_input_train_feature,np.array(target_feature_train_df)]
            test_array = np.c_[transformed_input_test_feature,np.array(target_feature_test_df)]

            save_numpy_array_data(array=train_array,file_path=self.data_transformation_config.transformed_train_file_path)
            save_numpy_array_data(array=test_array,file_path=self.data_transformation_config.transformed_test_file_path)
            save_object(obj=preprocessor_obj,file_path=self.data_transformation_config.transformed_object_file_path)

            ## preparing artifact 
            data_transformation_artifact:DataTransformationArtifact = DataTransformationArtifact(
                transformed_object_file_path  = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path  = self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)

        



