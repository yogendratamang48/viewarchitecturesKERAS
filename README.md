## Show Architecture in Keras
### Installations
* Make sure you have installed:
    * pydot `pip install pydot`
    * graphviz systemwise `sudo apt-get install graphviz`
    * If you are on windows, [go to here](https://graphviz.gitlab.io/download/) and add enviroment variable to bin directory if necessary.
    * graphviz in python `pip install graphviz`
### Saving and Loading Models
#### Imports
```python
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
```
#### Saving Model
After training, you can save model in json format. `json` format will only save the architecture.
```python
print("Saving Model")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print("Serializing and Saving")
model.save_weights("model.h5")
print("Model Saved")
```
#### Loading Model
```python
print("Loading json file")
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
```
### Creating Diagram
```python
from keras.utils import plot_model
plot_model(loaded_model, to_file='model.png', show_shapes=True,
           show_layer_names=True)
```
### Readings
1. https://keras.io/visualization/
2. https://machinelearningmastery.com/save-load-keras-deep-learning-models/ 
