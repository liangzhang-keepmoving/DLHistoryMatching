import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np
import glob




class CNN_HM():
    def __init__(self, dataset_path, label_file,geology_property):
        label_path=dataset_path+'/'+label_file
        self.dataframe = pd.read_csv(label_path, dtype=str)
        self.dataframe["read_images_address"] = dataset_path+'/'+self.dataframe["WELL"]+"_202_%s.png" % (geology_property)
        self.num_categories = self.dataframe['class_'].value_counts().shape[0]
        self.geology_property = geology_property

        # setup network
        self.cnn = tf.keras.models.Sequential()
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        self.cnn.add(tf.keras.layers.Flatten())
        self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.cnn.add(tf.keras.layers.Dense(units=self.num_categories, activation='softmax'))
        self.cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        #return self.cnn

    def load_petroleum_dataset(self):


        train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True,
                                    validation_split=0.2)
        training_set = train_datagen.flow_from_dataframe(
            self.dataframe,
            directory=None,
            x_col="read_images_address",
            y_col="class_",
            weight_col=None,
            target_size=(64, 64),
            color_mode="rgb",
            classes=None,
            class_mode="categorical",
            batch_size=32,
            shuffle=True,
            seed=None,
            save_to_dir=None,
            save_prefix="",
            save_format="png",
            subset="training",
            interpolation="nearest",
            validate_filenames=True)

        test_datagen = ImageDataGenerator(rescale = 1./255,validation_split=0.2)
        validating_set = test_datagen.flow_from_dataframe(
            self.dataframe,
            directory=None,
            x_col="read_images_address",
            y_col="class_",
            weight_col=None,
            target_size=(64, 64),
            color_mode="rgb",
            classes=None,
            class_mode="categorical",
            batch_size=32,
            shuffle=True,
            seed=None,
            save_to_dir=None,
            save_prefix="",
            save_format="png",
            subset="validation",
            interpolation="nearest",
            validate_filenames=True)
        
        return training_set, validating_set

    def train(self, epochs, save_location=None):
        #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        training_set, validating_set = self.load_petroleum_dataset()
        
        self.cnn.fit(x = training_set, validation_data=validating_set, epochs = epochs)
        if save_location:
            #self.cnn.save('path/to/location')
            self.cnn.save(save_location)


    def predict(self, prediction_samples, load_model=None):
        if load_model:
            model = tf.keras.models.load_model(load_model)
        else:
            model = self.cnn
        

        f_address = glob.glob(prediction_samples + '/'+'*.png')
        for i in range(len(f_address)):
            
            f_name = f_address[i].replace(prediction_samples+"/","")
            print("Sample", i+1, f_name)
            f_name = f_name.replace("_202_"+self.geology_property+".png","")
            
            test_image = tf.keras.preprocessing.image.load_img(f_address[i], target_size = (64, 64))
            test_image = image.img_to_array(test_image)/225
            test_image = np.expand_dims(test_image, axis = 0)
            result = model.predict(test_image)
            
            #print("The probability of each types:", result)
            print("The prediction result: type", np.argmax(result))
            self.dataframe.index = self.dataframe["WELL"]
            print("The result suppose to be: type", self.dataframe.loc[f_name,"class_"])
            print('\n')


