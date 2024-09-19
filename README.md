# Digit-Recognition-with-CNN accuracy 98%
 implements and explains Python code to recognize handwritten digits with CNN using Keras accuracy 98%
# Key Components
# 1-Environment Setup:
    The code begins by installing the Kaggle API and loading the dataset from Kaggle. 
    This includes authentication using a kaggle.json file, which contains the API credentials.
# 2-Data Loading:
    The training and test datasets are loaded into pandas DataFrames.
    The training dataset includes both images and labels, while the test dataset contains only images.
# 3-Data Preprocessing:
    The features and labels are separated from the training DataFrame.
    Checks for null values are performed on the datasets to ensure data integrity.
    The pixel values of the images are normalized to the range [0, 1] by dividing by 255.
    The images are reshaped to include a channel dimension, necessary for CNN input.
# 4-Data Visualization:
    A random selection of 25 images is displayed along with their corresponding labels to visualize the dataset.
# 5-Label Encoding:
    The labels are one-hot encoded using to_categorical,
    which is crucial for multi-class classification tasks.
# 6-Data Splitting:
    The training set is split into training and validation sets using train_test_split,
    allowing for better model evaluation.
# 7-Model Architecture:
    A CNN is defined with the following layers:
    Two convolutional layers followed by max pooling and dropout layers to reduce overfitting.
    Additional convolutional layers for deeper feature extraction.
    A flattening layer followed by fully connected (dense) layers.
    The output layer uses softmax activation to produce probabilities for the 10 digit classes.
# 8-Model Compilation:
    The model is compiled with the RMSprop optimizer and categorical cross-entropy loss, suitable for multi-class classification.
# 9-Data Augmentation:
    An ImageDataGenerator is employed to augment the training data, 
    introducing random transformations to enhance generalization and mitigate overfitting.
# 10-Model Training:
    The model is trained using the augmented data, with validation performed on the hold-out set. 
    The training process also incorporates learning rate reduction upon stagnation in validation accuracy.
# 11-Model Evaluation:
    After training, the model's performance is evaluated using accuracy and loss metrics, 
    and the results are plotted for both training and validation sets.
# 12-Confusion Matrix:
    A confusion matrix is generated to visualize the performance of the model across different classes. 
    This provides insight into specific misclassifications.
13-Prediction Function:
    A function is defined to predict and display the class of a given image. 
    It demonstrates the model's ability to classify unseen data.
# Results
The model achieves an accuracy of approximately 95.36% on the training data and 99.23% on the validation data, indicating strong performance.
The confusion matrix visually represents classification accuracy and areas of confusion, helping to identify digits that are frequently misclassified.
