
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
plt.switch_backend('WebAgg')


image_size = (224, 224)
batch_size = 8

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        # horizontal_flip=True
        )
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_ds = train_datagen.flow_from_directory(
        '/Autism/archive/train',
        target_size= image_size,
        batch_size= batch_size,
        class_mode='binary')

Valid_ds = test_datagen.flow_from_directory(
        '/Autism/archive/valid',
        target_size= image_size,
        batch_size=batch_size,
        class_mode='binary')

test_ds = test_datagen.flow_from_directory(
       'Autism/archive/test',
        target_size= image_size,
        batch_size=batch_size,
        class_mode='binary')

 #Model
def make_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Conv2D(32, 1, strides=1, padding="same", kernel_initializer=tf.keras.initializers.glorot_normal())(inputs)
    x = layers.SeparableConv2D(32,3,padding = 'same')(x)
    x = layers.BatchNormalization()(x) #224
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, 3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.glorot_normal())(x)
    x = layers.SeparableConv2D(64,3,padding = 'same')(x)
    x = layers.BatchNormalization()(x) #112
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(128, 3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.glorot_normal())(x)
    x = layers.SeparableConv2D(128,3,padding = 'same')(x)
    x = layers.BatchNormalization()(x) #56
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(256, 3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.glorot_normal())(x)
    x = layers.SeparableConv2D(256,3,padding = 'same')(x)
    x = layers.BatchNormalization()(x) #28
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(512, 3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.glorot_normal())(x)
    x = layers.SeparableConv2D(512,3,padding = 'same')(x)
    x = layers.BatchNormalization()(x) #14
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(1024, 3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.glorot_normal())(x)
    x = layers.SeparableConv2D(1024,3,padding = 'same')(x)
    x = layers.BatchNormalization()(x) #7
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(1024, 3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.glorot_normal())(x)
    x = layers.SeparableConv2D(1024,3,padding = 'same')(x)
    x = layers.BatchNormalization()(x) #7
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(512, 3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.glorot_normal())(x)
    x = layers.SeparableConv2D(512,3,padding = 'same')(x)
    x = layers.BatchNormalization()(x) #7
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(512, 3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.glorot_normal())(x)
    x = layers.SeparableConv2D(512,3,padding = 'same')(x)
    x = layers.BatchNormalization()(x) #7
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.GlobalMaxPool2D()(x) 
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(1,activation = 'sigmoid')(x)

    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,))
keras.utils.plot_model(model, show_shapes=True)

model.summary()
epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(
            train_ds, epochs=epochs, callbacks=callbacks, validation_data=Valid_ds, steps_per_epoch=5077 // batch_size, validation_steps=200 // batch_size
            )

plt.figure(figsize=(10,10))
plt.plot(history.history['loss'],label = "Training")
plt.plot(history.history['val_loss'],label = "Validation")
plt.title("weighted MSE loss trend")
plt.ylabel("MSE Value")
plt.xlabel("No. epoch")
plt.legend(loc = "upper left")
plt.show()
model.load_weights('save_at_50.h5')
evaluation= model.evaluate(test_ds,steps=200//batch_size)
print(evaluation)
