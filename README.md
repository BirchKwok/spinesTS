# spinesTS (spines for Time Series)
spinesTS is a Toolsets for time series.
## Install
```
pip install spinesTS
```
## Modules
```
├── spinesTS
│ ├── base
│ ├── data
│ ├── feature_extract
│ ├── metrics
│ ├── ml_model
│ ├── nn
│ ├── pipeline
│ ├── plotting
│ ├── preprocessing
│ └── utils
```
## Tutorials
```python
# simple demo to predict supermarket daily incoming
from spinesTS.pipeline import Pipeline
from spinesTS.data import LoadSupermarketIncoming
from spinesTS.ml_model import MultiOutputRegressor
from spinesTS.preprocessing import split_series
from spinesTS.plotting import plot2d
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt


# load data
data = LoadSupermarketIncoming()
df = data.dataset

# print(df.columns)
# print(df.head())

x_train, x_test, y_train, y_test = split_series(df['goods_cnt'], 
                                                df['goods_cnt'],
                                                # sliding window size, every 30 before days to predict after days
                                                window_size=30, 
                                                # predict after 30 days goods incoming
                                                pred_steps=30, 
                                                train_size=0.8
                                                )
print(f"x_train shape is {x_train.shape}, "
      f"x_test shape is {x_test.shape}," 
      f"y_train shape is {y_train.shape},"
      f"y_test shape is {y_test.shape}")

multi_reg = Pipeline([
    ('sc', StandardScaler()),
    ('model', MultiOutputRegressor(LGBMRegressor(random_state=2022)))
])
print("Model successfully initialization...")

# fit the model
multi_reg.fit(x_train, y_train, eval_set=(x_test, y_test))
print(f"r2_score is {multi_reg.score(x_test, y_test)}")

# plot the predicted results
plot2d(y_test, multi_reg.predict(x_test), figsize=(12, 8), 
       fig_num_or_slice=slice(-1, None), labels=['y_test', 'y_pred'])
plt.show()
```