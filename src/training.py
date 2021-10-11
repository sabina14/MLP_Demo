from src.utils.common_utils import read_config
from src.utils.model import create_model, saveplot,save_model
from src.utils.data_mgmt import get_data
import pandas as pd
import argparse
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "running_logs.log"), level=logging.INFO, format=logging_str,
                    filemode="a")


def training(config_path):
    config = read_config(config_path)
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
    NO_CLASSES = config["params"]["no_clasess"]
    OPTIMIZER = config["params"]["optimizer"]
    LOSS_FUNCTION = config["params"]["loss_function"]
    METRICS = config["params"]["metrics"]
    # print(OPTIMIZER)
    # print(NO_CLASSES)
    # print(LOSS_FUNCTION)
    # print(METRICS)
    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NO_CLASSES)

    EPOCHS = config["params"]["epochs"]

    VALIDATION_SET = (X_valid, y_valid)
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_SET)

    loss = history.history

##plot=history.history
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir=config["artifacts"]["model_dir"]
    model_dir_path=os.path.join(artifacts_dir,model_dir)
    os.makedirs(model_dir_path,exist_ok=True)
    model_name=config["artifacts"]["model_name"]
    save_model(model,model_name,model_dir_path)
    
    
    
    
    loss = history.history
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    plot_name = config["artifacts"]["plot_name"]
 
    plots_dir = config["artifacts"]["plots_dir"]
    plot_dir_path = os.path.join(artifacts_dir, plots_dir)
    os.makedirs(plot_dir_path,exist_ok=True)
    saveplot(loss, plot_name, plot_dir_path)


  

  


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>>>>>>>>>>STARTING TRAINING>>>>>>>>")
        training(config_path=parsed_args.config)
        logging.info("<<<<<<<<<<<<<<<<TRAINING DONE SUCCESSFULLY<<<<<<<<<<<<\n")

    except Exception as e:
        logging.exception(e)
        raise e
