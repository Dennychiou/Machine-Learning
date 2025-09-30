from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_data(cpu):
    cpu.target = cpu.target.values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(cpu.data, cpu.target, test_size=0.2, random_state=42)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    return x_tr, x_val, x_test, y_tr, y_val, y_test

def model(a, x_tr, x_val, y_tr, y_val, x_test, y_test, units):
    model = Sequential()
    model.add(Input((a,)))
    model.add(Dense(units, activation='sigmoid'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    history = model.fit(x_tr, y_tr, epochs=50, validation_data=(x_val, y_val), verbose=1)
    test_loss, test_mse = model.evaluate(x_test, y_test, verbose=1)
    return history, test_mse

def plot(history):
    plt.plot(history.history['mse'], label="Training")
    plt.plot(history.history['val_mse'], label="Validation")
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

def main():
    results = []
    cpu = fetch_openml(data_id=287)
    x_tr, x_val, x_test, y_tr, y_val, y_test = preprocess_data(cpu)
    history1, mse1 = model(11, x_tr, x_val, y_tr, y_val, x_test, y_test, 10)
    history2, mse2 = model(11, x_tr, x_val, y_tr, y_val, x_test, y_test, 100)
    history3, mse3 = model(11, x_tr, x_val, y_tr, y_val, x_test, y_test, 400)
    results.append(['Dataset 287', 'Model 1', mse1])
    results.append(['Dataset 287', 'Model 2', mse2])
    results.append(['Dataset 287', 'Model 3', mse3])
    plot(history1)
    plot(history2)
    plot(history3)
    return results

def main2():
    results = []
    cpu2 = fetch_openml(data_id=503)
    x_tr, x_val, x_test, y_tr, y_val, y_test = preprocess_data(cpu2)
    history1, mse1 = model(14, x_tr, x_val, y_tr, y_val, x_test, y_test, 10)
    history2, mse2 = model(14, x_tr, x_val, y_tr, y_val, x_test, y_test, 100)
    history3, mse3 = model(14, x_tr, x_val, y_tr, y_val, x_test, y_test, 400)
    results.append(['Dataset 503', 'Model 1', mse1])
    results.append(['Dataset 503', 'Model 2', mse2])
    results.append(['Dataset 503', 'Model 3', mse3])
    plot(history1)
    plot(history2)
    plot(history3)
    return results

results1 = main()
results2 = main2()

# Combine results and display them as a table
all_results = results1 + results2
df_results = pd.DataFrame(all_results, columns=['Dataset', 'Model', 'Test MSE'])
print(df_results)
