import os
import cv2
import numpy as np
from keras import layers 
from keras  import models
from keras import utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

classes=["Acta-De-Nacimiento","Carnet-De-Seguro-Medico","Cartilla-Militar","Credencial-Universitaria","Curp","Ine",
         "Licencia-De-Conducir","Pasaporte","Rfc","Tarjeta-De-Credito","Tarjeta-De-Membresia","Tarjeta-De-Metro","Visa"]

num_classes = len(classes)
img_rows, img_cols = 64, 64

def load_data():
    data = [] 
    target = []
    
    for index, clase in enumerate(classes):
        folder_path = os.path.join('src/data/training', clase)
        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (img_rows, img_cols))
            data.append(np.array(image))
            target.append(index)
    
    data = np.array(data)
    data = data.reshape(data.shape[0], img_rows, img_cols, 1)
    target = np.array(target)
    
    new_target = utils.to_categorical(target, num_classes)
    return data, new_target


data, target = load_data()
print(data)

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax')) 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=40, epochs=100, verbose=1, validation_data=(x_test, y_test))

model.save('src/model/document-model.h5')

if not os.path.exists('graphics/matrix'):
    os.makedirs('graphics/matrix')
    os.makedirs('graphics/history')


y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
confusion_matrix = confusion_matrix(y_true, y_pred_classes) 

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicción')
plt.ylabel('Valor real')
plt.savefig('graphics/matrix/matrix_confusion_document.png')
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
print(history.history['loss'])
print(history.history['val_loss'])
plt.title('History of error')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
plt.savefig('graphics/history/history_document.png')
plt.show()