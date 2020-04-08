import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle


def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)*ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__ == "__main__":
    df = pd.read_csv('insurance.csv')
    #Data is splited into train and test category
    train, test = data_split(df, 0.2)

    # Split the input in X factor
    x_train = train[['age','sex','bmi','children','smoker','region']].to_numpy()
    x_test = test[['age','sex','bmi','children','smoker','region']].to_numpy()

    # Split the output in Y factor
    y_train = train[['charges']].to_numpy().reshape(1071,-1)
    y_test = test[['charges']].to_numpy().reshape(267,-1)

    # Linear model Build
    reg = linear_model.LinearRegression()

    # Provide Data to Model
    reg.fit(x_train, y_train)


    modelPrediction = reg.predict(x_test)
    
    print('Mean squared error: %.2f'
      % mean_squared_error(y_test, modelPrediction))

    ##The coefficient of determination: 1 is perfect prediction

    print('Coefficient of determination: %.2f'
        % r2_score(y_test, modelPrediction))


    # Predict the Value from the model
    modelPrediction = reg.predict([[19,0,27.9,0,1,1]])
    print("Result",modelPrediction)

    # open a file, where you want to store the data
    file = open('model.pkl','wb')

    
    # dump information to that file
    pickle.dump(reg, file)

    # close the file
    file.close()

    
