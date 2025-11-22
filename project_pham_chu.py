import librosa as lb
import numpy as np
import os
import matplotlib.pyplot as plt
import noisereduce as nr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score


# Create dataset for the project
# for each recorded signal:
#    if the signal is longer than 5 second, trim the first or last part
#   depends on the sum of that part
#    if the signal is shorter than 5 second, append 0s to the beginning 
#   of the signal until it reaches required length
sr = 48000
length_expected = int(sr * 5)
data_folder = 'data/'
tram = []
car = []
for o in ['car', 'tram']:
    data_files = data_folder + o + '/'
    for name in os.listdir(data_files):
        path = os.path.join(data_files, name)
        if path.__contains__('.wav'):
            data, sr = lb.load(path, sr=None)

            # data cleansing
            length_data = len(data)
            if length_data > length_expected:
                remain = int(length_data - length_expected)
                if data[:remain].sum() > data[-remain:].sum():
                    data = data[:length_expected].copy()
                else:
                    data = data[-length_expected:].copy()

            elif length_data < length_expected:
                remain = int(length_expected - length_data)
                added_noise = np.zeros((remain,))
                data = np.concat([added_noise, data]).copy()
            if path.__contains__('tram'):
                tram.append(data)
            else:
                car.append(data)

tram = np.array(tram)
car = np.array(car)


# Combine dataset
x = np.vstack((tram, car))
x = nr.reduce_noise(x, sr)

# stft: each signal's stft is calculated then concatenated to form a long sample
period = 0.1
win_size = int(sr * period)
window = np.hamming(win_size)
nfft = win_size
hop_size = int(0.5 * win_size)

x_stft = np.abs(lb.stft(x, n_fft=nfft, win_length=win_size, hop_length=hop_size, window=window)).reshape((57,-1))
for i in range(len(x_stft)):
    x_stft[i,:] = (x_stft[i,:] - np.mean(x_stft[i,:])) / np.std(x_stft[i,:])

# Define labels
y_tram = np.zeros(tram.shape[0]) # 27 zeros for tram
y_car = np.ones(car.shape[0]) # 30 ones for car
y = np.concatenate((y_tram, y_car))

x_train, x_test, y_train, y_test = train_test_split(x_stft, y, random_state=16, test_size = 0.2)


# Using Cross Validation to Get the Best Value of k
k_values = [i for i in range (1,31)]
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, x, y, cv=5, scoring='f1', n_jobs=-1)
    scores.append(np.mean(score))

# sns.lineplot(x = k_values, y = scores, marker = 'o')
plt.plot(k_values,scores)
plt.xlabel("k values")
plt.ylabel("f1 score")

best_index = np.argmax(scores)
best_k = k_values[best_index]


knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")


# test with a sample
pred_sample = y_pred[0]
true_sample = y_test[0]

print(f'for sample x_test[0] {x_test[0]}')
print(f'predicted value: {pred_sample}')
print(f'truth value: {true_sample}')

# plt.show()