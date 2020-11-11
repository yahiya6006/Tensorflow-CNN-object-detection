#------- Importing Libraries ------------
import cv2
import tensorflow as tf
import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt

#------- Variables to store the data --------------
image_data = []
cordinates = []
filenames = []
print("[INFO] Processing csv file")

#------- Opening the csv file -----------------
data_csv = open("car.csv").read().strip().split("\n")
for data in data_csv[1:len(data_csv)]:
    data = data.split(",")
    (filename,width,height,cls,xmin,ymin,xmax,ymax) = data

    #---------- Read image to find height and width ---------------
    img = cv2.imread("data/"+filename)
    (h, w) = img.shape[:2]

    #------- Scalling cordinates data wrt height and width --------
    x = float(xmin) / w
    y = float(ymin) / h
    w = float(xmax) / w
    H = float(ymax) / h
    
    #------- Loading image with set target size and preprocess --------
    img = tf.keras.preprocessing.image.load_img("data/"+filename, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)

    #------ Adding the data to list --------
    image_data.append(img)
    cordinates.append((x,y,w,H))
    filenames.append(filename)

#----- Normalizing and converting to numpy array ----------
image_data = np.array(image_data, dtype="float32") / 255.0
cordinates = np.array(cordinates, dtype="float32")

#----- using sklearn to split the data intp 90% : 10% --------
(x_train, x_test, y_train, y_test, x_filenames, y_filenames) = sklearn.model_selection.train_test_split(image_data, cordinates,
                                                 filenames, test_size=0.10,random_state=42)

f = open("Testfiles.txt","w")
f.write("\n".join(y_filenames))
f.close()
#----- Loading resnet50 and doing transfer learning -------
resnet50 = tf.keras.applications.ResNet50V2(weights="imagenet", include_top=False,
                                        input_shape=(224, 224, 3))
resnet50.trainable = False

output = resnet50.output
flatten = tf.keras.layers.Flatten()(output)
x = tf.keras.layers.Dense(128, activation="relu")(flatten)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dense(4, activation="sigmoid")(x)

model = tf.keras.Model(inputs=resnet50.input, outputs=x)
print(model.summary())

model.compile(optimizer=tf.optimizers.Adam(lr=0.00001),
              loss = 'mse',
              metrics=['accuracy']
              )

#------ Creating callback to cancel training when reached its accuracy ------
class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        self.model.save("car.h5",save_format="h5")
        if(logs.get('accuracy')>0.95):
            print('\nReached 95% cancelling training')
            self.model.stop_training = True
callback = mycallback()
            
print("[INFO] Starting training")

H = model.fit(x_train,
                    y_train,
                    validation_data=(x_test, y_test),
                    batch_size=4,
                    epochs=900,
                    verbose=1,
                    callbacks=[callback])

model.save("car.h5",save_format="h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("plot.png")

