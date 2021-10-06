from utils.common_utils import read_config
from utils.model import create_model
from utils.data_mgmt import get_data

import argparse


def training(config_path):
    config=read_config(config_path)
    validation_datasize=config["params"]["validation_datasize"]
    (X_train,y_train),(X_valid,y_valid),(X_test,y_test)=get_data(validation_datasize)
    NO_CLASSES=config["params"]["no_clasess"]
    OPTIMIZER=config["params"]["optimizer"]
    print(OPTIMIZER)
    LOSS_FUNCTION=config["params"]["loss_function"]
    METRICS=config["params"]["metrics"]
    model=create_model(LOSS_FUNCTION,OPTIMIZER,METRICS,NO_CLASSES)

    EPOCHS=config["params"]["epochs"]
    
    VALIDATION_SET=(X_valid,y_valid)
    history=model.fit(X_train,y_train,epochs=EPOCHS,validation_data=VALIDATION_SET)


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)