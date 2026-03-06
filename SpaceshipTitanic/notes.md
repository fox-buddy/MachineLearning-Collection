# Keras Training Time on Colab

| Environment | Model Type | Epochs| Training Time |
|-------------|------------|-------|---------------|
|CPU|Sequential Single Model|60|222 sec|
|GPU|Sequential Single Model|60|11 sec|
|GPU|Sequential Random Search|60|11 sec|

Performed 20 Random Search Runs with the following Random Search Parameters

```
import keras_tuner as kt

def build_model(hp):
  # First Part Definition of hyper Parameters
  # Hidden neurons
  n_hidden = hp.Int("n_hidden", min_value=0, max_value=10, default=3)

  # Neuron Count
  n_neurons = hp.Int("n_neurons", min_value=16, max_value=128)

  # Learning Rate
  learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-1, sampling="log")

  # We could also define some special parameters per Optimizer. For Example momentum for sgd

  # optimizer
  optimizer = hp.Choice("optimizer", values=["sgd", "adam", "adamw"])

  if optimizer == "sgd":
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  elif optimizer == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  else:
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)

  
  # Seconde Part - Build Model
  model = tf.keras.Sequential()
  
  # model.add(tf.keras.layers.Flatten())  # Only multi dimensional Data

  # Add hidden layers
  for _ in range(n_hidden):
    model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))

  # Add output layer
  model.add(tf.keras.layers.Dense(1, activation="sigmoid")) 


  model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

  return model
```



# Accuracy Advanced Data (Age Group - 6248 Features)
- Random Forest: 0,78
- Voting Classifier: 0,77
- Sequential Keras model: 0,801 (31, 31, 31)

# Accuracy Minimal Data (Numerical Scaling - 24 Features)
- Random Forest: 0,80
- SciKit MLP: 0,78
- SVM: 0,77
- Voting Classifier: 0,79

# Age Group Data (31 Features)
