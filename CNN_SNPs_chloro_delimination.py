import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

infile=open("trainingset.1.chloro.txt","r")

sequences = infile.read().split('\n')
sequences = list(filter(None, sequences))  # This removes empty sequences.

infile.close()

# Let's print the first few sequences.
pd.DataFrame(sequences, index=np.arange(1, len(sequences)+1), 
             columns=['Sequences']).head()

# The LabelEncoder encodes a sequence of bases as a sequence of integers.
integer_encoder = LabelEncoder()  
 
# The OneHotEncoder converts an array of integers to a sparse matrix where 
# each row corresponds to one possible value of each feature.
one_hot_encoder = OneHotEncoder(categories='auto')   

input_features = []

for sequence in sequences:
  integer_encoded = integer_encoder.fit_transform(list(sequence))
  integer_encoded = np.array(integer_encoded).reshape(-1, 1)
  one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
  input_features.append(one_hot_encoded.toarray())

np.set_printoptions(threshold=40)
input_features = np.stack(input_features)
print("Example sequence\n-----------------------")
print('DNA Sequence #1:\n',sequences[0][:10],'...',sequences[0][-10:])
print('One hot encoding of Sequence #1:\n',input_features[0].T)

labels_infile=open("trainingset.name.chloro.txt","r")
labels = labels_infile.read().split('\n')
labels = list(filter(None, labels))  # removes empty sequences

one_hot_encoder = OneHotEncoder(categories='auto')
labels = np.array(labels).reshape(-1, 1)
input_labels = one_hot_encoder.fit_transform(labels).toarray()

print('Labels:\n',labels.T)
print('One-hot encoded labels:\n',input_labels.T)


train_features, test_features, train_labels, test_labels = train_test_split(
    input_features, input_labels, test_size=0.25, random_state=42)


tf.keras.backend.clear_session()
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=12, 
                 input_shape=(train_features.shape[1], 4)))
model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(filters=32, kernel_size=12, 
                 input_shape=(train_features.shape[1], 4)))
model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5,restore_best_weights = True)

history = model.fit(input_features, input_labels, 
                    epochs=50, verbose=1, validation_split=0.25)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

###interpret###

infile=open("testingset.1.chloro.txt","r")

sequences2 = infile.read().split('\n')
sequences2 = list(filter(None, sequences2))  # This removes empty sequences.

infile.close()


pd.DataFrame(sequences2, index=np.arange(1, len(sequences)+1), 
             columns=['Sequences']).head()

integer_encoder2 = LabelEncoder()

one_hot_encoder2 = OneHotEncoder(categories='auto')   
input_features2 = []

for sequence2 in sequences2:
  integer_encoded2 = integer_encoder.fit_transform(list(sequence2))
  integer_encoded2 = np.array(integer_encoded2).reshape(-1, 1)
  one_hot_encoded2 = one_hot_encoder.fit_transform(integer_encoded2)
  input_features2.append(one_hot_encoded2.toarray())

#np.set_printoptions(threshold=40)
input_features2 = np.stack(input_features2)
print("Example sequence\n-----------------------")
print('DNA Sequence #1:\n',sequences2[0][:10],'...',sequences2[0][-10:])
print('One hot encoding of Sequence #1:\n',input_features2[0].T)

predictions = model(input_features2)

class_names=['CS','LS','LYS','MXG','SS']

inname=open("testingset.name.chloro.txt","r")

sequences3 = inname.read().split('\n')

samples = sequences3
k=-1
for i, logits in enumerate(predictions):
  k += 1
  class_idx = tf.argmax(logits).numpy()
  c0=logits[0]
  c1=logits[1]
  c2=logits[2]
  c3=logits[3]
  c4=logits[4]
  p0 = tf.nn.softmax(logits)[0]
  p1 = tf.nn.softmax(logits)[1]
  p2 = tf.nn.softmax(logits)[2]
  p3 = tf.nn.softmax(logits)[3]
  p4 = tf.nn.softmax(logits)[4]
  print("Example {} prediction: {} ({:4.1f}%),  {} ({:4.1f}%),  {} ({:4.1f}%), {} ({:4.1f}%),{} ({:4.1f}%),sample: {}".format(i+1, class_names[0],100*p0,class_names[1],100*p1,class_names[2],100*p2,class_names[3],100*p3,class_names[4],100*p4,samples[k]))
  #print("Example {} prediction: {} ({:4.1f}%),  {} ({:4.1f}%),  {} ({:4.1f}%), {} ({:4.1f}%),{} ({:4.1f}%)sample: {}".format(i+1, class_names[0],100*c0,class_names[1],100*c1,class_names[2],100*c2,class_names[3],100*c3,class_names[4],100*c4,samples[k]))
