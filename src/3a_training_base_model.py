import pandas as pd
import numpy as np
import joblib
import time

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def create_linreg_model(X_train, y_train, X_test, y_test):
  base_model = LinearRegression()
  start_time = time.time()
  base_model.fit(X_train, y_train)
  elapsed_time = (time.time() - start_time)

  y_pred = base_model.predict(X_test)

  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)
  aard = 100*np.mean(np.abs(y_pred - y_test) / y_test)
  r2 = r2_score(y_test, y_pred)

  print("base_model coef(s):", base_model.coef_)
  print("Mean Squared Error:", mse)
  print("RMSE:", rmse)
  print("AARD:", aard)
  print("R² Score:", r2)
  print(f'Total training time: {elapsed_time/60:.0f} min {elapsed_time%60:.2f} sec')
  
  return base_model

if __name__=='__main__':
  X_train = joblib.load('pickles/X_train.pkl')
  y_train = joblib.load('pickles/y_train.pkl')
  X_test = joblib.load('pickles/X_test.pkl')
  y_test = joblib.load('pickles/y_test.pkl')

  linreg_model = create_linreg_model(X_train, y_train, X_test, y_test)
  joblib.dump(linreg_model, 'pickles/model_linreg.pkl')
