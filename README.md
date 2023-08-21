# spinesTS 
## Time Series forecasting toolsets

- [Install](https://github.com/BirchKwok/spinesTS#install)
- [spinesTS Modules](https://github.com/BirchKwok/spinesTS#spinests-modules)
- [Tutorials](https://github.com/BirchKwok/spinesTS#tutorials)
  - [Getting started](https://github.com/BirchKwok/spinesTS#getting-started)
  - [Using nn module](https://github.com/BirchKwok/spinesTS#using-nn-module)
    - [StackingRNN](https://github.com/BirchKwok/spinesTS#stackingrnn)
    - [GAUNet](https://github.com/BirchKwok/spinesTS#gaunet)
    - [Time2VecNet](https://github.com/BirchKwok/spinesTS#time2vecnet)
  - [Using ml_model module](https://github.com/BirchKwok/spinesTS#using-ml_model-module)
    - [MultiStepRegressor](https://github.com/BirchKwok/spinesTS#multistepregressor)
    - [MultiOutputRegressor](https://github.com/BirchKwok/spinesTS#multioutputregressor)
    - [WideGBRT](https://github.com/BirchKwok/spinesTS#widegbrt)
  - [Using Data module](https://github.com/BirchKwok/spinesTS#using-data-module)


## Install
```
pip install spinesTS
```

## spinesTS Modules

- base: Model base class
- data: Built-in datasets and data wrapper classes
- feature_generator: Feature generation functions
- metrics: Model performance measurement function
- ml_model: Machine learning models
- nn: neural network models
- pipeline: Model fitting and prediction pipeline
- plotting: Visualization of model prediction results
- preprocessing: data preprocessing
- utils: Tool functions set
- layers: Neural network layer

## Tutorials

### Getting started
```python
# simple demo to predict Electric data
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

from spinesTS.pipeline import Pipeline
from spinesTS.data import LoadElectricDataSets
from spinesTS.ml_model import MultiOutputRegressor
from spinesTS.preprocessing import split_series
from spinesTS.plotting import plot2d


# load data
df = LoadElectricDataSets()

# split data
x_train, x_test, y_train, y_test = split_series(
    x_seq=df['value'], 
    y_seq=df['value'],  # The sequence of parameter y_seq is cut based on parameter x_seq
    # sliding window size, every 30 before days to predict after days
    window_size=30, 
    # predict after 30 days
    pred_steps=30, 
    train_size=0.8
)

print(f"x_train shape is {x_train.shape}, "
      f"x_test shape is {x_test.shape}," 
      f"y_train shape is {y_train.shape},"
      f"y_test shape is {y_test.shape}")

# Assemble the model using Pipeline class
model = Pipeline([
    ('sc', StandardScaler()),
    ('model', MultiOutputRegressor(LGBMRegressor(random_state=2022)))
])
print("Model successfully initialization...")

# fitting model
model.fit(x_train, y_train, eval_set=(x_test, y_test), verbose=0)
print(f"r2_score is {model.score(x_test, y_test)}")

# plot the predicted results
fig = plot2d(y_test, model.predict(x_test), figsize=(20, 10), 
       eval_slices='[:30]', labels=['y_test', 'y_pred'])
plt.show()
```
```
[output]:
x_train shape is (270, 30), x_test shape is (68, 30),y_train shape is (270, 30),y_test shape is (68, 30)
Model successfully initialization...
r2_score is 0.8186046606725977
```
![model prediction image](https://github.com/BirchKwok/spinesTS/blob/main/examples/visual/GettingStarted.png)

### Using nn module
#### StackingRNN
```python
import matplotlib.pyplot as plt

from spinesTS.data import LoadElectricDataSets
from spinesTS.preprocessing import split_series
from spinesTS.plotting import plot2d
from spinesTS.nn import StackingRNN
from spinesTS.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error


# load data
df = LoadElectricDataSets()

# split data
x_train, x_test, y_train, y_test = split_series(
    x_seq=df['value'], 
    y_seq=df['value'],
    # sliding window size, every 128 before days to predict after days
    window_size=128, 
    # predict after 24 days goods incoming
    pred_steps=24, 
    train_size=0.8
)

print(f"x_train shape is {x_train.shape}, "
      f"x_test shape is {x_test.shape}," 
      f"y_train shape is {y_train.shape},"
      f"y_test shape is {y_test.shape}")

# model initialization
model = StackingRNN(in_features=128, out_features=24, 
                    random_seed=42, loss_fn='mae', 
                    learning_rate=0.001, dropout=0.1, diff_n=1, 
                    stack_num=2, bidirectional=True, device='cpu')

model.fit(x_train, y_train, eval_set=(x_test[:-2], y_test[:-2]), batch_size=32,
             min_delta=0, patience=100, epochs=3000, verbose=False, lr_scheduler=None)
y_pred_cs = model.predict(x_test[-2:])
print(f"r2: {r2_score(y_test[-2:].T, y_pred_cs.T)}")
print(f"mae: {mean_absolute_error(y_test[-2:], y_pred_cs)}")
print(f"mape: {mean_absolute_percentage_error(y_test[-2:], y_pred_cs)}")
a = plot2d(y_test[-2:], y_pred_cs, eval_slices='[-1]', labels=['y_test', 'y_pred'], figsize=(20, 6))
plt.show()
```
#### GAUNet
```python
import matplotlib.pyplot as plt

from spinesTS.data import LoadElectricDataSets
from spinesTS.preprocessing import split_series
from spinesTS.plotting import plot2d
from spinesTS.nn import GAUNet
from spinesTS.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error


# load data
df = LoadElectricDataSets()

# split data
x_train, x_test, y_train, y_test = split_series(
    x_seq=df['value'], 
    y_seq=df['value'],
    # sliding window size, every 128 before days to predict after days
    window_size=128, 
    # predict after 24 days 
    pred_steps=24, 
    train_size=0.8
)

print(f"x_train shape is {x_train.shape}, "
      f"x_test shape is {x_test.shape}," 
      f"y_train shape is {y_train.shape},"
      f"y_test shape is {y_test.shape}")

# model initialization
model = GAUNet(in_features=128, out_features=24, 
               random_seed=42, flip_features=False, 
               learning_rate=0.001, level=5, device='cpu')

model.fit(x_train, y_train, eval_set=(x_test[:-2], y_test[:-2]), batch_size=32,
             min_delta=0, patience=100, epochs=3000, verbose=False, lr_scheduler='ReduceLROnPlateau')
y_pred_cs = model.predict(x_test[-2:])
print(f"r2: {r2_score(y_test[-2:].T, y_pred_cs.T)}")
print(f"mae: {mean_absolute_error(y_test[-2:], y_pred_cs)}")
print(f"mape: {mean_absolute_percentage_error(y_test[-2:], y_pred_cs)}")
a = plot2d(y_test[-2:], y_pred_cs, eval_slices='[-1]', labels=['y_test', 'y_pred'], figsize=(20, 6))
plt.show()
```
#### Time2VecNet
```python
import matplotlib.pyplot as plt

from spinesTS.data import LoadElectricDataSets
from spinesTS.preprocessing import split_series
from spinesTS.plotting import plot2d
from spinesTS.nn import Time2VecNet
from spinesTS.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error


# load data
df = LoadElectricDataSets()

# split data
x_train, x_test, y_train, y_test = split_series(
    x_seq=df['value'], 
    y_seq=df['value'],
    # sliding window size, every 128 before days to predict after days
    window_size=128, 
    # predict after 24 days 
    pred_steps=24, 
    train_size=0.8
)

print(f"x_train shape is {x_train.shape}, "
      f"x_test shape is {x_test.shape}," 
      f"y_train shape is {y_train.shape},"
      f"y_test shape is {y_test.shape}")

# model initialization
model = Time2VecNet(in_features=128, out_features=24, 
               random_seed=42, flip_features=False, 
               learning_rate=0.001, device='cpu')

model.fit(x_train, y_train, eval_set=(x_test[:-2], y_test[:-2]), batch_size=32,
             min_delta=0, patience=100, epochs=3000, verbose=False, lr_scheduler='CosineAnnealingLR')
y_pred_cs = model.predict(x_test[-2:])
print(f"r2: {r2_score(y_test[-2:].T, y_pred_cs.T)}")
print(f"mae: {mean_absolute_error(y_test[-2:], y_pred_cs)}")
print(f"mape: {mean_absolute_percentage_error(y_test[-2:], y_pred_cs)}")
a = plot2d(y_test[-2:], y_pred_cs, eval_slices='[-1]', labels=['y_test', 'y_pred'], figsize=(20, 6))
plt.show()
```

### Using ml_model module
#### MultiStepRegressor
```python
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

from spinesTS.data import LoadElectricDataSets
from spinesTS.ml_model import MultiStepRegressor
from spinesTS.preprocessing import split_series
from spinesTS.plotting import plot2d


# load data
df = LoadElectricDataSets()

# split data
x_train, x_test, y_train, y_test = split_series(
    df['value'], 
    df['value'],
    # sliding window size, every 30 before days to predict after days
    window_size=30, 
    # predict after 30 days 
    pred_steps=30, 
    train_size=0.8
)

print(f"x_train shape is {x_train.shape}, "
      f"x_test shape is {x_test.shape}," 
      f"y_train shape is {y_train.shape},"
      f"y_test shape is {y_test.shape}")

# model initialization
model = MultiStepRegressor(LGBMRegressor(random_state=2022))
print("Model successfully initialization...")

# fitting model
model.fit(x_train, y_train, eval_set=(x_test, y_test), verbose=0)
print(f"r2_score is {model.score(x_test, y_test)}")

# plot the predicted results
fig = plot2d(y_test, model.predict(x_test), figsize=(20, 10), 
       eval_slices='[:30]', labels=['y_test', 'y_pred'])
plt.show()
```
#### MultiOutputRegressor
```python
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

from spinesTS.data import LoadElectricDataSets
from spinesTS.ml_model import MultiOutputRegressor
from spinesTS.preprocessing import split_series
from spinesTS.plotting import plot2d


# load data
df = LoadElectricDataSets()

# split data
x_train, x_test, y_train, y_test = split_series(
    df['value'], 
    df['value'],
    # sliding window size, every 30 before days to predict after days
    window_size=30, 
    # predict after 30 days 
    pred_steps=30, 
    train_size=0.8
)

print(f"x_train shape is {x_train.shape}, "
      f"x_test shape is {x_test.shape}," 
      f"y_train shape is {y_train.shape},"
      f"y_test shape is {y_test.shape}")

# model initialization
model = MultiOutputRegressor(LGBMRegressor(random_state=2022))
print("Model successfully initialization...")

# fitting model
model.fit(x_train, y_train, eval_set=(x_test, y_test), verbose=0)
print(f"r2_score is {model.score(x_test, y_test)}")

# plot the predicted results
fig = plot2d(y_test, model.predict(x_test), figsize=(20, 10), 
       eval_slices='[:30]', labels=['y_test', 'y_pred'])
plt.show()
```
#### WideGBRT
```python
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

from spinesTS.data import LoadElectricDataSets
from spinesTS.ml_model import GBRTPreprocessing, WideGBRT
from spinesTS.plotting import plot2d


# load data
df = LoadElectricDataSets()

# split data and generate new features
gbrt_processor = GBRTPreprocessing(in_features=128, out_features=30, 
                                   target_col='value', train_size=0.8, date_col='date',
                                   differential_n=1  # The order of data differentiation.
                                   )
gbrt_processor.fit(df)

x_train, x_test, y_train, y_test = gbrt_processor.transform(df)

print(f"x_train shape is {x_train.shape}, "
      f"x_test shape is {x_test.shape}," 
      f"y_train shape is {y_train.shape},"
      f"y_test shape is {y_test.shape}")

# model initialization
model = WideGBRT(model=LGBMRegressor(random_state=2022))
print("Model successfully initialization...")

# fitting model
model.fit(x_train, y_train, eval_set=(x_test, y_test), verbose=0)
print(f"r2_score is {model.score(x_test, y_test)}")

# plot the predicted results
fig = plot2d(y_test, model.predict(x_test), figsize=(20, 10), 
       eval_slices='[:30]', labels=['y_test', 'y_pred'])
plt.show()
```

### Using Data module
```python
from spinesTS.data import *
series_data = BuiltInSeriesData(print_file_list=True)
```
```
+---+----------------------+----------------------------------------------+
|   | ds name              | columns                                      |
+---+----------------------+----------------------------------------------+
| 0 | ETTh1                | date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT |
| 1 | ETTh2                | date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT |
| 2 | ETTm1                | date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT |
| 3 | ETTm2                | date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT |
| 4 | Electric_Production  | date, value                                  |
| 5 | Messages_Sent        | date, ta, tb, tc                             |
| 6 | Messages_Sent_Hour   | date, hour, ta, tb, tc                       |
| 7 | Supermarket_Incoming | date, goods_cnt                              |
| 8 | Web_Sales            | date, type_a, type_b, sales_cnt              |
+---+----------------------+----------------------------------------------+
```
```python
# select one dataset
df_a = series_data['ETTh1']  # series_data[0], it works, too
print(type(df_a))  # <class 'spinesTS.data._data_base.DataTS'>

# Because DataTS inherit from pandas DataFrame, it has all the functionality of pandas DataFrame
df_a.head() ,df_a.tail(), df_a.shape
```

