from sklearn.svm import SVR
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


def load_data(filepath=r"data.csv", isTraining=True, days=7):
    data = np.array(pd.read_csv(filepath,encoding="utf-8"))[:,1]
    if isTraining:
        data = data[:int(data.shape[0]*0.7)]
    else:
        data = data[int(data.shape[0]*0.7):]
    x_data, y_data = [], []
    for index in range(data.shape[0]-days):
        x_data.append(data[index:(index+days)])
        y_data.append(data[index+days])
    x_data = np.array(x_data).astype(float)
    y_data = np.array(y_data).astype(float)
    return x_data, y_data

def train(x_data, y_data):
    clf = SVR(kernel='linear', C=1.0, epsilon=0.2)
    model = clf.fit(x_data, y_data)
    return model

def test(x_data, model):
    return model.predict(x_data)

def plot_graph(y_test, y_pred):
    # plt.figure()
    # plt.style.use('dark_background')
    plt.plot(y_pred, color='r', label=r"y_pred")
    plt.plot(y_test, color='c', label=r"y_true")
    plt.legend(loc='upper left')
    plt.show()

def losses(y_pred, y_true):
    RMSE = np.sqrt(np.mean((y_true-y_pred)**2))
    MAE = np.mean(np.abs(y_true-y_pred))
    MAPE = np.mean(np.abs(y_pred-y_true)/y_true)*100
    print(
      "RMSE:{}\n".format(RMSE),
      "MAE:{}\n".format(MAE),
      "MAPE:{}".format(MAPE)
    )
    

def main():
    x_data, y_data = load_data()
    model = train(x_data, y_data)
    x_test, y_test = load_data(isTraining=False)
    y_pred = test(x_test, model)
    losses(y_pred, y_test)
    plot_graph(y_test, y_pred)

if __name__ == "__main__":
    main()