
"""
Edit 16.11 - może działa ale na mniejszej ilości danych
"""


import binance
import matplotlib.pyplot as plt
from binance import Client
import pandas as pd
from tensorflow import keras
import numpy as np
from keras import backend as K
client = Client()

frame = pd.read_csv("data.txt")
frame = pd.DataFrame(frame)
frame = frame.iloc[:,1:]
frame.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
frame = frame.astype(float)

n = 2
DATA_RANGE = 120
END = n*24*60
LABEL_LEN = int(0.2 * (END - DATA_RANGE))
pod = 1

BTC_ACC = 0.0
USDT_ACC = 1000.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(DATA_RANGE, 5)),  #input- flatten-po prostu z 2 wymiarowej tablicy robi 1 wymiarow
    keras.layers.Dense(DATA_RANGE*5, activation="sigmoid"),  #hidden - actiwation to po prostu funkcja zwikszajaca zlozonosc algorytmu
    keras.layers.Dense(DATA_RANGE, activation="sigmoid"),
    keras.layers.Dense(DATA_RANGE/30, activation="sigmoid"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adamax", loss="sparse_categorial_crossentropy", metrics=["accuracy", 'mse'])



for k in range(pod):
    print("Round: ", k+1)
    rows = []
    train_data = []
    train_labels = []

    print('starting datatsets')
    for a in range(k*int((END-DATA_RANGE)/pod), (k+1)*int((END-DATA_RANGE)/pod)):
        # creating train data
        rows = []
        for i in range(a, DATA_RANGE + a):
            rows.append([float(frame['High'][i]), float(frame['Open'][i]), float(frame['Low'][i]), float(frame['Close'][i]), float(frame['Volume'][i])])
        train_data.append(rows)

        # creating train labels
        train_labels.append(frame['Close'][a + DATA_RANGE])
        """
        if frame['Close'][a + DATA_RANGE] > 0:
            train_labels.append(1)
        else:
            train_labels.append(0)
        """


    print("done")
    print("starting training model")
    model.fit(train_data, train_labels, epochs=100)


#saving model
model.save("model[10d_2h_1m_e100].h5")


print("creating test data")
test_data = []
test_labels = []

for a in range(LABEL_LEN):
    # creating test data
    rows = []
    for i in range(a+END + 1, DATA_RANGE + END +1 + a):
        rows.append([float(frame['High'][i]), float(frame['Open'][i]), float(frame['Low'][i]), float(frame['Close'][i]), float(frame['Volume'][i])])
    test_data.append(rows)

    # creating train labels
    if frame['Close'][a + DATA_RANGE] > 0:
        test_labels.append(1)
    else:
        test_labels.append(0)



print(test_labels, len(test_labels), len(test_data))
"""
np.save("test_data", test_data)
np.save("test_labels", test_labels)

test_data = []
test_labels = []

train_data = []
train_labels = []
test_data = np.load("test_data.npy")
test_labels = np.load("test_labels.npy")
"""
test_loss, test_acc, test_mse = model.evaluate(test_data, test_labels)
print("Tested loss: ", test_loss)
print("Tested acc: ", test_acc)
print("Tested mse: ", test_mse)

acc = 0
lost = 0
output = model.predict(test_data)
for i in range(len(output)):
    if output[i] > 0.5:
        output[i] = 1
    else:
        output[i] = 0

    
results = pd.DataFrame({'y': np.array(test_labels).flatten(), 'y_pred': np.array(output).flatten()})
results.plot(title="model pred vs act", figsize=(17,7))
plt.show()

"""
frame1 = pd.DataFrame(client.get_historical_klines('BTCUSDT','1m', '1 days ago UTC'))
frame1 = frame1.iloc[:,1:6]
frame1.columns = ['Open', 'High', 'Low','Close', 'Volume']
frame1 = frame1.astype(float)
L = len(frame1)

row = []
input=[]

print("creatin data")
for j in range(0, L - DATA_RANGE-1):
    for i in range(j, DATA_RANGE+j):
        row.append([float(frame1['High'][i]), float(frame1['Open'][i]), float(frame1['Low'][i]), float(frame1['Close'][i]), float(frame1['Volume'][i])])
    input.append(row)
    row=[]


print("predict time")
output = model.predict(input)
print(output)

"""
"""
print("cash cash cash")
for i in range(len(output)):
    if output[i] == 0:
        BTC_ACC += USDT_ACC/frame1['Close'][i]
        USDT_ACC = 0
    else:
        USDT_ACC += BTC_ACC*frame1['Close'][i]
        BTC_ACC = 0

print("BTC: ", BTC_ACC, "USDT: ", USDT_ACC, "btc in usdt: ", BTC_ACC*float(frame1['Close'][8*24*60-1]))
"""






