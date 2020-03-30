## Code taken from the DonkeyCar project
## Thanks to W. Roscoe and the DonkeyCar community

def model_categorical(input_size= (90,250,3), dropout=0.1):
    '''Generate an NVIDIA AutoPilot architecture.
    Input_size: Image shape (90, 250, 3), adjust to your desired input.
    Dropout: Proportion of dropout used to avoid model overfitting.
    This model ONLY predicts steering angle as a 5-elements array encoded with a Softmax output.
    The model is already compiled and ready to be trained.
    '''
    import keras
    from keras.layers import Input, Dense, merge
    from keras.models import Model
    from keras.layers import Conv2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Dense

    """
    Input layer instantiates a tensor with the Input() function which contains
    # the information and takes as parameters the size of the images taken by the camera.
    """
    img_in = Input(shape=input_size, name='img_in')

    # Input
    x = img_in
    """
    2nd to 6th layer : Conv2D
    For those layers, we do 2D convolutions with the Conv2D function.
    This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
    The first parameter corresponds to filters: the dimensionality of the output space (i.e. the number of output filters in the convolution).
    The next two parameters the kernel size: it specifies the height and width of the 2D convolution window.
    The Subsample (i.e. strides) parameter specify the strides of the convolution along the height and width.
    The relu activation function arguments makes that all the negative values returned in the tensor will take the value 0.
    """
    x = Conv2D(24, 5,5, subsample=(2,2), activation='relu')(x)       # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Conv2D(32, 5,5, subsample=(2,2), activation='relu')(x)       # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Conv2D(64, 5,5, subsample=(2,2), activation='relu')(x)       # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Conv2D(64, 3,3, subsample=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Conv2D(64, 3,3, subsample=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    # 7th layer: flatten the input and reduces the dimension of the tensor.
    x = Flatten(name='flattened')(x)                                        # Flatten to 1D (Fully connected)
    # Classify the data into 100 features, make all negatives 0
    x = Dense(100, activation='relu')(x)
    # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dropout(dropout)(x)
    # Classify the data into 50 features, make all negatives 0
    x = Dense(50, activation='relu')(x)
    # Randomly drop out 10% of the neurons (Prevent overfitting)
    x = Dropout(dropout)(x)

    # Categorical output of the angle
    # Data classified into 5 features. Use a softmax activation which gives the probability for an image to belong to each category.
    angle_out = Dense(5, activation='softmax', name='angle_out')(x)

    # The model takes Img_in as input and angle_out as an output.
    model = Model(input=[img_in], output=[angle_out])

    """
    Compile the model with adam for optimizer which is more efficient.
    Categorical crossentropy is a loss function that is used for single label categorization.
    This is when only one category is applicable for each data point. In other words, an example can belong to one class only.
    Accuracy returns the percentage of correct predictions.
    """
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
