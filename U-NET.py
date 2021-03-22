import tensorflow as tf
import random
import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt
import pickle
import time


#zdefiniowanie potrzebnych metryk
def jaccard_index_numpy(y_true, y_pred):
    TP = np.count_nonzero(y_pred * y_true)
    FP = np.count_nonzero(y_pred * (y_true - 1))
    FN = np.count_nonzero((y_pred - 1) * y_true)

    if (TP + FP + FN) == 0:
        jac = 0
    else:
        jac = TP / (TP + FP + FN)

    return jac

def dice_score(y_true, y_pred):
    TP = np.count_nonzero(y_pred * y_true)
    FP = np.count_nonzero(y_pred * (y_true - 1))
    FN = np.count_nonzero((y_pred - 1) * y_true)
    if (TP + FP + FN) == 0:
        dice = 0
    else:
        dice = 2*TP / (2*TP + FP + FN)
    return dice

def accuracy(y_true, y_pred):
    TP = np.count_nonzero(y_pred * y_true)
    FP = np.count_nonzero(y_pred * (y_true - 1))
    FN = np.count_nonzero((y_pred - 1) * y_true)
    TN = np.count_nonzero((y_pred - 1) * (y_true - 1))
    if (TP + FP + FN) == 0:
        accuracy = 0
    else:
        accuracy = (TP+TN) / (TP+TN+FP+FN)
    return accuracy

#nazwa sieci
NAME = "U-Net-Drone-256p-BS16 n -32{}".format(int(time.time()))



gpus = tf.config.experimental.list_physical_devices('GPU')  
tf.config.experimental.set_virtual_device_configuration(
          gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])




IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

seed = 42
np.random.seed = seed

pickle_in = open("X_train256.pickle","rb")
X_train = pickle.load(pickle_in)

pickle_in = open("Y_train256.pickle","rb")
Y_train = pickle.load(pickle_in)

pickle_in = open("X_test256.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("Y_test256.pickle","rb")
Y_test = pickle.load(pickle_in)

inputs = tf.keras.layers.Input((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS))

### Architektura sieci 
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Sciezka enkodera
c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Sciezka dekodera
u6 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)


outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
# NAME2 = "U-NET finall model"
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.save(NAME)
# model.save(NAME2)

###### checkpointy i wyniki

BATCH_SIZE = 16
patience = 5
checpointer = tf.keras.callbacks.ModelCheckpoint('model_drone.h5', verbose=1, save_best_only=True)
csv_logger = tf.keras.callbacks.CSVLogger('log_unet.csv', append=True, separator=';')
callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NAME),
                          histogram_freq=1, 
                          write_graph=True, 
                          write_grads=True, 
                          batch_size=BATCH_SIZE, 
                          write_images=True)]

results = model.fit(X_train,Y_train, 
                    validation_split=0.2, 
                    batch_size=BATCH_SIZE,
                    epochs=100,
                    shuffle=True,
                    callbacks=[callbacks] )


n_epochs = len(results.history['loss'])
min_loss = min(results.history['loss'])
min_val_loss = min(results.history['val_loss'])
formatted_float = "{:.3f}".format(min_val_loss)
print(results.history)
epochs_list = []
for n in range(1,n_epochs+1):
    epochs_list.append(n)
X=n_epochs-patience
Y=min_val_loss
y_1=results.history['loss']
y_2=results.history['val_loss']

 #wizualizacja krzywych uczenia
fig = plt.figure(figsize=(60, 30))
# plt.plot(results.history['loss'], linewidth=8, color='r')           
# plt.plot(results.history['val_loss'], linewidth=8, color='b')
plt.plot(epochs_list,y_1, linewidth=8,color='g')
plt.plot(epochs_list,y_2,linewidth=8,color='b')
plt.plot(X,Y, marker='o', markerfacecolor='red', markersize=32)
plt.annotate('min_val_loss = {}'.format(formatted_float), 
                 (X,Y),
                 fontsize=40,
                 textcoords="offset pixels", 
                 xytext=(0,70), 
                 ha='center') 
plt.title('Train and Validation Loss', fontsize=100, fontweight="bold")
plt.ylabel('Loss', fontsize=80)
plt.xlabel('Epoch', fontsize=80)
plt.legend(['Train', 'Validation'], loc='upper right', fontsize=50)
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.show()
fig.savefig("loss-{}.jpg".format(NAME))

###############

idx = random.randint(0, len(X_train))


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)
 

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

score = jaccard_index_numpy(Y_test, preds_test_t)
score_dice = dice_score(Y_test, preds_test_t)
accuracy = accuracy(Y_test, preds_test_t)

# sprawdzenie dzialania na kilku obrazach 
ix = random.randint(0, len(preds_train_t))
fig, ax1 = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.5))
# wyswietlenie oryginalu
ax1[0].imshow(X_train[ix])
ax1[0].set_title('Obraz oryginalny')
ax1[0].axis('off')

#wyswietlenie maski
ax1[1].imshow(np.squeeze(Y_train[ix]), cmap='gray' )
ax1[1].set_title('Maska obrazu')
ax1[1].axis('off')

#wyswietlenie wyniku
ax1[2].imshow(preds_train_t[ix], cmap='gray' )
ax1[2].set_title('Wynik segmentacji')
ax1[2].axis('off')

plt.imsave("wyniki/train_org1.jpg",X_train[ix])

plt.imsave("wyniki/mask_org1.jpg",np.squeeze(Y_train[ix]))

plt.imsave("wyniki/siectrening_org1.jpg",np.squeeze(preds_train_t[ix]))




ix3 = random.randint(0, len(preds_val_t))
fig, ax2 = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.5))
# wyswietlenie oryginalu
ax2[0].imshow(X_train[int(X_train.shape[0]*0.9):][ix3])
ax2[0].set_title('Obraz oryginalny')
ax2[0].axis('off')

#wyswietlenie maski
ax2[1].imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix3]), cmap='gray' )
ax2[1].set_title('Maska obrazu')
ax2[1].axis('off')

#wyswietlenie wyniku
ax2[2].imshow(np.squeeze(preds_val_t[ix3]), cmap='gray' )
ax2[2].set_title('Wynik segmentacji')
ax2[2].axis('off')


plt.imsave("wyniki/val_org1.jpg",X_train[int(X_train.shape[0]*0.9):][ix3])

plt.imsave("wyniki/valmask_org1.jpg",np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix3]))

plt.imsave("wyniki/valwyn_org1.jpg",np.squeeze(preds_val_t[ix3]))



ix2 = random.randint(0, len(preds_test_t))
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.5))
# wyswietlenie oryginalu
ax[0].imshow(X_test[ix2])
ax[0].set_title('Obraz oryginalny')
ax[0].axis('off')

#wyswietlenie maski
ax[1].imshow(np.squeeze(Y_test[ix2]), cmap='gray' )
ax[1].set_title('Maska obrazu')
ax[1].axis('off')

#wyswietlenie wyniku
ax[2].imshow(np.squeeze(np.squeeze(preds_test_t[ix2])), cmap='gray' )
ax[2].set_title('Wynik segmentacji')
ax[2].axis('off')

plt.imsave("wyniki/testorg1.jpg",X_test[ix2])

plt.imsave("wyniki/mask_test1.jpg",np.squeeze(Y_test[ix2]))

plt.imsave("wyniki/siectest_org1.jpg",np.squeeze(preds_test_t[ix2]))



 






