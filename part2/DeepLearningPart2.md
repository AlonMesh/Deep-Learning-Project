```python
# Import TensorFlow 2.x in compatibility mode
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

    WARNING:tensorflow:From C:\Users\Alon\AppData\Roaming\Python\Python38\site-packages\tensorflow\python\compat\v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    


```python
import pandas as pd
```


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
```

## Load and preprocess the dataset


```python
data = pd.read_csv(r'C:\Users\Alon\OneDrive\שולחן העבודה\קורסים\למידה עמוקה\archive\diabetes.csv')
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   768 non-null    int64  
     2   BloodPressure             768 non-null    int64  
     3   SkinThickness             768 non-null    int64  
     4   Insulin                   768 non-null    int64  
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    int64  
     8   Outcome                   768 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB
    

## Define features and labels


```python
y = data['Outcome']
```


```python
y
```




    0      1
    1      0
    2      1
    3      0
    4      1
          ..
    763    0
    764    0
    765    0
    766    1
    767    0
    Name: Outcome, Length: 768, dtype: int64




```python
X = data.copy().drop(columns='Outcome')
```


```python
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>763</th>
      <td>10</td>
      <td>101</td>
      <td>76</td>
      <td>48</td>
      <td>180</td>
      <td>32.9</td>
      <td>0.171</td>
      <td>63</td>
    </tr>
    <tr>
      <th>764</th>
      <td>2</td>
      <td>122</td>
      <td>70</td>
      <td>27</td>
      <td>0</td>
      <td>36.8</td>
      <td>0.340</td>
      <td>27</td>
    </tr>
    <tr>
      <th>765</th>
      <td>5</td>
      <td>121</td>
      <td>72</td>
      <td>23</td>
      <td>112</td>
      <td>26.2</td>
      <td>0.245</td>
      <td>30</td>
    </tr>
    <tr>
      <th>766</th>
      <td>1</td>
      <td>126</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>30.1</td>
      <td>0.349</td>
      <td>47</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1</td>
      <td>93</td>
      <td>70</td>
      <td>31</td>
      <td>0</td>
      <td>30.4</td>
      <td>0.315</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 8 columns</p>
</div>



## Split the data into train and test sets


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

# Neural Network Model

## Define the neural network architecture


```python
input_size = X_train.shape[1]
output_size = 1  # Simple classification
```


```python
# Placeholder for input data
X = tf.placeholder(tf.float32, shape=[None, input_size], name='X')
```


```python
# Placeholder for labels
y = tf.placeholder(tf.float32, shape=[None, output_size], name='y')
```


```python
# Define the neural network model
hidden_size = 32

# Define the number of hidden layers (L)
L = 3

# Input layer
hidden_layer = X

# Create L hidden layers
for i in range(L):
    hidden_layer = tf.layers.dense(inputs=hidden_layer, units=hidden_size, activation=tf.nn.relu, name=f'hidden_layer_{i+1}')

# Output layer
output_layer = tf.layers.dense(inputs=hidden_layer, units=output_size, name='output_layer')
```

    C:\Users\Alon\AppData\Local\Temp\ipykernel_6172\719582209.py:12: UserWarning: `tf.layers.dense` is deprecated and will be removed in a future version. Please use `tf.keras.layers.Dense` instead.
      hidden_layer = tf.layers.dense(inputs=hidden_layer, units=hidden_size, activation=tf.nn.relu, name=f'hidden_layer_{i+1}')
    C:\Users\Alon\AppData\Local\Temp\ipykernel_6172\719582209.py:15: UserWarning: `tf.layers.dense` is deprecated and will be removed in a future version. Please use `tf.keras.layers.Dense` instead.
      output_layer = tf.layers.dense(inputs=hidden_layer, units=output_size, name='output_layer')
    


```python
# Define loss function and optimizer
loss = tf.reduce_mean(tf.square(output_layer - y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)
```


```python
# Initialize variables
init = tf.global_variables_initializer()
```

## Train the model


```python
epochs = 5000
batch_size = 32

# Lists to store training history
training_losses_nn = []

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size].values.reshape(-1, 1)  # Convert to NumPy array and reshape

            sess.run(train_op, feed_dict={X: X_batch, y: y_batch})

        # Print training progress
        if (epoch + 1) % 500 == 0:
            training_loss = sess.run(loss, feed_dict={X: X_train, y: y_train.values.reshape(-1, 1)})
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {training_loss}')
            
            # Save training loss for plotting
            training_losses_nn.append(training_loss)

    # Make predictions
    predictions = sess.run(output_layer, feed_dict={X: X_train})
```

    Epoch 500/5000, Training Loss: 0.14503219723701477
    Epoch 1000/5000, Training Loss: 0.08713172376155853
    Epoch 1500/5000, Training Loss: 0.05831257998943329
    Epoch 2000/5000, Training Loss: 0.04374520108103752
    Epoch 2500/5000, Training Loss: 0.047553613781929016
    Epoch 3000/5000, Training Loss: 0.0443120151758194
    Epoch 3500/5000, Training Loss: 0.040675923228263855
    Epoch 4000/5000, Training Loss: 0.04373455420136452
    Epoch 4500/5000, Training Loss: 0.020810794085264206
    Epoch 5000/5000, Training Loss: 0.021850351244211197
    


```python
plt.plot(range(500, epochs + 1, 500), training_losses_nn, marker='o', linestyle='-', color='b')
plt.grid()
plt.title('Training Loss over Epochs (Neural Network)')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.show()
```


    
![png](output_23_0.png)
    


# Comparison to a simple logistic regression 

## Define the logistic regression model parameters


```python
weights = tf.Variable(tf.zeros([input_size, 1]), name='weights')
bias = tf.Variable(tf.zeros([1]), name='bias')
```


```python
# Define the logistic regression model
input_size = X_train.shape[1]

# Placeholder for input data
X = tf.placeholder(tf.float32, shape=[None, input_size], name='X')

# Placeholder for binary labels (0 or 1)
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
```


```python
# Logistic regression model
logits = tf.add(tf.matmul(X, weights), bias)
predictions = tf.nn.sigmoid(logits)

# Define loss function (binary cross-entropy)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

# Define optimizer (e.g., gradient descent)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)
```


```python
# Initialize variables
init = tf.global_variables_initializer()
```


```python
# Train the logistic regression model
epochs = 5000
batch_size = 32

# List to store training history
training_losses_log_r = []

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size].values.reshape(-1, 1)  # Convert to NumPy array and reshape

            sess.run(train_op, feed_dict={X: X_batch, y: y_batch})

        # Print training progress
        if (epoch + 1) % 500 == 0:
            training_loss = sess.run(loss, feed_dict={X: X_train, y: y_train.values.reshape(-1, 1)})
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {training_loss}')

            # Save training loss for plotting
            training_losses_log_r.append(training_loss)
```

    Epoch 500/5000, Training Loss: 30.284317016601562
    Epoch 1000/5000, Training Loss: 11.173237800598145
    Epoch 1500/5000, Training Loss: 8.017516136169434
    Epoch 2000/5000, Training Loss: 10.407505989074707
    Epoch 2500/5000, Training Loss: 7.838350296020508
    Epoch 3000/5000, Training Loss: 25.219282150268555
    Epoch 3500/5000, Training Loss: 23.743389129638672
    Epoch 4000/5000, Training Loss: 11.036818504333496
    Epoch 4500/5000, Training Loss: 26.197052001953125
    Epoch 5000/5000, Training Loss: 6.833233833312988
    


```python
plt.plot(range(500, epochs + 1, 500), training_losses_log_r, marker='o', linestyle='-', color='g')
plt.grid()
plt.title('Training Loss over Epochs (Logistic Regression)')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.show()
```


    
![png](output_31_0.png)
    


## Comparison in a plot


```python
plt.plot(range(500, epochs + 1, 500), training_losses_nn, marker='o', linestyle='-', color='b', label='Neural Network')
plt.plot(range(500, epochs + 1, 500), training_losses_log_r, marker='o', linestyle='-', color='g', label='Logistic Regression')

# Adding labels and title
plt.grid()
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')

plt.legend()  # Adding legend

plt.show()
```


    
![png](output_33_0.png)
    

