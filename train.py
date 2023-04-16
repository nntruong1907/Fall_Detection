from datetime import datetime
import itertools
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import pandas as pd
from def_lib import load_data

train_path = './data/train_5f.csv'
test_path = './data/test_5f.csv'

# Load the train data
X, y, class_names = load_data(train_path)
# Split training data (X, y) into (X_train, y_train) and (X_val, y_val)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
# Load the test data
X_test, y_test, _ = load_data(test_path)
# Pre-process data
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[3],X_train.shape[1]))
X_val = np.reshape(X_val,(X_val.shape[0],X_val.shape[3],X_val.shape[1]))
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[3],X_test.shape[1]))
print( "X_train:", X_train.shape)
print( "y_train:", y_train.shape)
print( "\nX_val:", X_val.shape)
print( "y_val:", y_val.shape)
print( "\nX_test:", X_test.shape)
print( "y_test:", y_test.shape)
print( "\nclass_names:", class_names)

# Define the model
def LSTM():
    inputs = tf.keras.Input(shape=(34, 5))
    layer = keras.layers.LSTM(
        32, activation=tf.nn.relu6, return_sequences=True)(inputs)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.LSTM(32, activation=tf.nn.relu6)(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.Dense(16, activation=tf.nn.relu6)(layer)
    layer = keras.layers.Dropout(0.2)(layer)

    outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

    model = keras.Model(inputs, outputs)

    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# def LSTM():
#     inputs = tf.keras.Input(shape=(34, 5))
#     layer = keras.layers.LSTM(
#         32, activation=tf.nn.relu6, return_sequences=True)(inputs)
#     layer = keras.layers.Dropout(0.2)(layer)
#     layer = keras.layers.LSTM(40, activation=tf.nn.relu6)(layer)
#     layer = keras.layers.Dropout(0.2)(layer)
#     layer = keras.layers.Dense(20, activation=tf.nn.relu6)(layer)
#     layer = keras.layers.Dropout(0.2)(layer)

#     outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

#     model = keras.Model(inputs, outputs)

#     model.summary()
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])

#     return model

#@title Make Model
model = LSTM()

# write Classification_report and Confusion_matrix to file
def plot_confusion_matrix(plot_confusion_matrix_path,cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """Plots the confusion matrix."""
        if normalize:
          cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
          print("Normalized confusion matrix")
        else:
          print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=55)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(plot_confusion_matrix_path)
        plt.close()

result_path = './results/'
model_path = os.path.join(result_path,'model_fall.h5')

checkpoint_path = os.path.join(result_path,"weights.best.hdf5")
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                            monitor='val_accuracy',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max')
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                            patience=20)

# Start training
print('---------------------------------- TRAINING -------------------------------------------')
start = datetime.now()
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=64,
                    validation_data=(X_val, y_val),
                    callbacks=[checkpoint, earlystopping])
# Save model
model.save(model_path)
print('--------------------------------- EVALUATION -----------------------------------------')
loss_test, accuracy_test = model.evaluate(X_test, y_test)
print('LOSS TEST: ', loss_test)
print("ACCURACY TEST: ", accuracy_test)
loss_train, accuracy_train = model.evaluate(X_train, y_train)
duration = datetime.now() - start

print('LOSS TRAIN: ', loss_train)
print("ACCURACY TRAIN: ", accuracy_train)
print('TIME COMPLETED: ', duration)

data_eval = 'LOSS TEST: ' + str(loss_test) + ' / ACCURACY TEST: ' + str(accuracy_test) \
            + '\n' + 'LOSS TRAIN: ' + str(loss_train) + ' / ACCURACY TRAIN: ' + str(accuracy_train) \
            + '\n' + 'TIME COMPLETED: ' + str(duration)

'''--------------------------------------- STATISTC -------------------------------------------'''
hist_df = pd.DataFrame(history.history)
name_history = 'history.csv'
path_history = os.path.join(result_path , name_history)
with open(path_history, mode='w') as f:
    hist_df.to_csv(f)

eval_path = 'Evaluation_LSTM_5f.txt'
path_s = os.path.join(result_path , eval_path)
with open(path_s, mode='w') as f:
    f.writelines(data_eval)
f.close()

model_path_load = os.path.join(result_path,'model_fall.h5')
plot_confusion_matrix_nor_path = os.path.join(result_path,'confusion_matrix_nor.png')
plot_confusion_matrix_path = os.path.join(result_path,'confusion_matrix.png')
model = tf.keras.models.load_model(model_path_load)

y_pred = model.predict(X_test)
# Convert the prediction result to class name
y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
y_true_label = [class_names[i] for i in np.argmax(y_test, axis=1)]

# Plot the confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
plot_confusion_matrix(plot_confusion_matrix_path,cm,
                        class_names,
                        title ='Confusion Matrix of Fall Detection Model')

plot_confusion_matrix(plot_confusion_matrix_nor_path,cm,
                    class_names,normalize=True,
                    title ='Normalized Confusion Matrix of Fall Detection Model')

# Print the classification report
print('\nClassification Report:\n', classification_report(y_true_label,
                                                            y_pred_label))


Classification_Report = os.path.join(result_path,'Classification_Report.txt')
with open(Classification_Report, mode='w') as f:
    f.writelines(classification_report(y_true_label,y_pred_label))
f.close()

# Visualize the training history to see whether you're overfitting.
image_acc_path = os.path.join(result_path,'model_acc_LSTM.png')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='best')
# plt.show()
plt.savefig(image_acc_path)
plt.close()

# Visualize the training history to see whether you're overfitting.
image_loss_path = os.path.join(result_path,'model_loss_LSTM.png')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='best')
# plt.show()
plt.savefig(image_loss_path)
plt.close()