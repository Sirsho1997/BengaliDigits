# Handwritten digit recognition for the fifth largest spoken language in the world

**CHECK THIS -->** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/BookGenrePrediction.ipynb)

The data set that has been used is CMATERdb data set. [https://code.google.com/archive/p/cmaterdb/] It is a balanced data set of total 6000 Bangla numerals.

CMATERdb is the pattern recognition database repository created at the 'Center for Microprocessor Applications for Training Education and Research' (CMATER) research laboratory, Jadavpur University, Kolkata 700032, INDIA. This database is free for all non-commercial uses.

It is a balanced dataset of total 6000 Bangla numerals (32x32 RGB coloured, 6000 images), each having 600 images per class (per digit).

  ## One data set example
  
  
 <img src="https://github.com/Sirsho1997/BengaliDigits/blob/master/image/trainingExample.png" width="25%" height="25%" />

  ## Building the model
  
  Building the neural network requires configuring the layers of the model, then compiling the model. The basic building block of a neural network is the layer. Layers extract representations from the data fed into them. Hopefully, these representations are meaningful for the problem at hand.

  ### CNN Architecture

A very common architecture for a CNN is a stack of Conv2D and MaxPooling2D layers followed by a few denesly connected layers. The idea is that the stack of convolutional and maxPooling layers extract the features from the image. Then these features are flattened and fed to densly connected layers that determine the class of an image based on the presence of features.

```python
     #Set up the layers
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))
```
  ## Compiling the model
  
  Before the model is ready for training, it needs a few more settings.

  Loss function —It measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
  
  Optimizer —It decides how the model is updated based on the data it sees and its loss function.

  Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
  
  ```python
     #Compile the model
      model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
```

  ## Evaluate Accuracy
  Next, comparing how the model performs on the test dataset
  
  ```python
     test_loss, test_accuracy = model.evaluate(test_x,  test_y, verbose=2)
print("Accuracy : ",test_accuracy*100,"%")
```
  <img src="https://github.com/Sirsho1997/BengaliDigits/blob/master/image/accuracy.png" width="50%" height="30%" />


  ## Plotting several images along with their predictions
  
  <img src="https://github.com/Sirsho1997/BengaliDigits/blob/master/image/prediction.png" width="100%" height="100%" />


Contributor - 
- [Sirshendu Ganguly](https://www.linkedin.com/in/sirshendu-ganguly/)  [![Github](https://github.com/Sirsho1997/BengaliDigits/blob/master/image/github.png](https://github.com/Sirsho1997)
