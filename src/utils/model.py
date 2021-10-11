import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time
import os

def create_model(LOSS_FUNCTION,OPTIMIZER,METRICS,NO_CLASSES):
    
  LAYERS=[

        tf.keras.layers.Flatten(input_shape=[28,28],name="inputLayer"),
        tf.keras.layers.Dense(300,activation="relu",name="hiddenLayer1"),
        tf.keras.layers.Dense(100,activation="relu",name="hiddenLayer2"),
        tf.keras.layers.Dense(NO_CLASSES,activation="softmax",name="outputLayer")
  ]
  model_clf=tf.keras.models.Sequential(LAYERS)
  model_clf.summary()
  #print(LOSS_FUNCTION)
  #print(OPTIMIZER)
  #print(METRICS)
  model_clf.compile(loss=LOSS_FUNCTION,optimizer= OPTIMIZER,metrics=METRICS)
  return model_clf




def get_unique_filename(file_name):
  unique_filename=time.strftime(f"%Y%m%d_%H%M%S_{file_name}")
  return unique_filename

def save_model(model,model_name,model_dir):
  unique_filename=get_unique_filename(model_name)
  path_to_model=os.path.join(model_dir,unique_filename)
  model.save(path_to_model)

def get_unique_plotname(plotname):
    unique_plotname=time.strftime(f"%Y%m%d_%H%M%S_{plotname}")
    return unique_plotname


def saveplot(loss,plot_name,plots_dir):
    pd.DataFrame(loss).plot(figsize=(10, 7))
    plt.grid(True)
    ##fig=plot.get_figure()
    unique_plotname=get_unique_plotname(plot_name)
    path_to_plot=os.path.join(plots_dir,unique_plotname)
    plt.savefig(path_to_plot)
    #plt.show()




