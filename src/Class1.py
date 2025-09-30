from keras.datasets import fashion_mnist
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

class FashionMNIST: 
    def __init__(self):
        (self.trainX, self.trainy), (self.testX, self.testy) = fashion_mnist.load_data()
        print('Train: X = ', self.trainX.shape)
        print('Test: X = ', self.testX.shape)

    def visualize(self, p=9):
        for i in range(1, p+1):
            plt.subplot(3, 3, i)
            plt.imshow(self.trainX[i], cmap="gray") 
        plt.show()

class DataPreprocessing:
    def __init__(self, trainX, testX):
        self.trainX = np.expand_dims(trainX, -1)
        self.testX = np.expand_dims(testX, -1)
        print("Train Shape (Preprocessed):", self.trainX.shape)

    def data(self):
        return self.trainX, self.testX

class ModelArch: 
    def __init__(self):
        self.model = Sequential()
    
    def model_build(self):
        self.model.add(Conv2D(64, (5, 5), padding="same", activation = "relu", input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, (5, 5), padding="same", activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256, (5, 5), padding="same", activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dense(10, activation="softmax"))
        return self.model

class ModelTraining:
    def __init__(self, model):
        self.model = model
        self.history = None
    def compile(self):
        self.model.compile(optimizer=Adam(learning_rate=1e-3),
                           loss="sparse_categorical_crossentropy",
                           metrics=["sparse_categorical_accuracy"])

    def train(self, trainX, trainy, epochs=10, validation_split=0.33, steps_per_epoch=100):
        self.history = self.model.fit(
            trainX.astype(np.float32),
            trainy.astype(np.float32),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_split=validation_split
        )
        return self.history

    def save_model(self, path="./model.h5"):
        self.model.save_weights(path, overwrite=True)

    def plot_history(self):
        # Accuracy
        plt.plot(self.history.history["sparse_categorical_accuracy"])
        plt.plot(self.history.history["val_sparse_categorical_accuracy"])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Val"], loc="upper left")
        plt.show()

        # Loss
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Val"], loc="upper left")
        plt.show()


class Network:
    def __init__(self, trainy):
        self.labels = [str(label) for label in np.unique(trainy)]

    def predict_and_visualize(self, model, testX):
        predictions = model.predict(testX[:1])
        label_index = np.argmax(predictions)
        label = self.labels[label_index]
        print("Predicted class index:", label_index)
        print("Predicted label:", label)
        plt.imshow(testX[:1][0])
        plt.show()



if __name__ == "__main__":

    dataset = FashionMNIST()
    dataset.visualize_samples()

    preprocess = DataPreprocessing(dataset.trainX, dataset.testX)
    trainX, testX = preprocess.get_data()

    model_arch = ModelArch()
    model = model_arch.build_model()
    model.summary()

    trainer = ModelTraining(model)
    trainer.compile()
    history = trainer.train(trainX, dataset.trainy)
    trainer.save_model()
    trainer.plot_history()

    network = Network(dataset.trainy)
    network.predict_and_visualize(model, testX)
