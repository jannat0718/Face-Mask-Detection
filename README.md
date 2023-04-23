## Face Mask Detection Using Deep Learning

**Introduction:**

In the era of the COVID-19 pandemic, wearing a face mask has become essential to reduce the spread of the virus. Automated face mask detection systems can help enforce mask-wearing rules in public spaces. This project aims to develop a deep learning model for detecting whether a person is wearing a mask or not using images. The model uses MobileNetV2 as the base model for transfer learning and applies image data augmentation techniques to improve performance. Additionally, we implement the YOLO v4 object detection model for face mask detection on images.

**Objective:**

The project's main objective is to develop a deep learning-based face mask detection system that accurately identifies whether individuals are wearing masks, assisting in monitoring and ensuring compliance with public health regulations. To achieve this, in the first part, the project will utilize a dataset of images, employ MobileNetV2 with transfer learning techniques to create the face mask classifier, and implement a data augmentation pipeline to enhance the classifier's generalization capabilities. In the second part, YOLOv4 object detection model implemented to detect if a person is wearing mask or not in the image.

**MobileNetV2:**

MobileNetV2 is a lightweight and efficient deep learning model primarily designed for mobile and embedded vision applications. It is an improvement over the original MobileNet model, focusing on reducing computational complexity while maintaining high accuracy levels. MobileNetV2 uses depthwise separable convolutions, a technique that significantly reduces the number of parameters and computations required when compared to traditional convolutional layers.

The key innovation in MobileNetV2 is the introduction of the inverted residual structure with linear bottlenecks. This architecture starts with a low-dimensional representation and expands it through a series of layers, followed by depthwise convolution and projection back to the lower-dimensional representation. This approach allows MobileNetV2 to retain more information through the network while minimizing computational costs. Furthermore, MobileNetV2 employs shortcut connections between bottlenecks, which aids in mitigating the vanishing gradient problem and improves overall performance.

MobileNetV2 is well-suited for various computer vision tasks, including image classification, object detection, and semantic segmentation, due to its high accuracy and low computational complexity. Its lightweight nature makes it an ideal choice for deploying on devices with limited resources, such as smartphones or edge devices.

**YOLOv4:**

YOLOv4 (You Only Look Once version 4) is a state-of-the-art real-time object detection model that builds on the principles of previous YOLO models. It combines novel techniques, such as Bag of Freebies (BoF) and Bag of Specials (BoS), with several architecture improvements, such as CSPDarknet53 as the backbone, PANet as the neck, and enhanced anchor boxes. YOLOv4 is designed to be faster and more accurate than its predecessors, while maintaining real-time detection capabilities. It works by dividing an input image into a grid and predicting bounding boxes and class probabilities for each grid cell simultaneously.

## **Step-by-step process:**

**First Part: Building a Face Mask Classifier**

1. The code begins by installing the required Keras library and importing necessary libraries like TensorFlow, Keras, layers, and so on.

2. A working directory and data directory are defined. The dataset has been manually downloaded from Kaggle and contains 3,833 images.

3. The code then unzips the dataset and stores it in the working directory.

4. Hyperparameters like batch size, learning rate, image height, and image width are defined.

5. Data generators for training and validation are created using the ImageDataGenerator function from Keras. This function applies various augmentation techniques like random zoom, rotation, shifts, and flips on the images to increase the dataset's size and improve the model's generalization capability. Train set and validation set split by 80%:20% ratio and 3067 images belonging to the training set and 766 in the validation set for 2 classes.

6. MobileNetV2 model is used as a base model. Its final layers are replaced with custom layers to adapt it to the binary classification task for face mask detection.

7. The newly created model is saved before compiling it.

8. The model is then compiled using the Adam optimizer, Binary Cross Entropy loss, and accuracy metric.

9. Early stopping and CSV logging callbacks are defined.

10. The model was trained using the model.fit method, with the training and validation data for 10 epochs. By stopping at early stopping round 9, the training was halted before overfitting could occur. Once training is completed, the model's performance is visualized using loss and accuracy plots. The training history indicated that the model was performing well on both training and validation sets. The decreasing loss and increasing accuracy with each epoch were good signs that the model was learning effectively from the data. The high validation accuracy suggested that the model generalized well to new, unseen data without overfitting the training data.

11. The new model, with loaded weights, is then saved as a complete face mask detection model for future use, such as evaluation or deployment in a real-time detection system.

12. The test dataset was downloaded using the Bing Image Downloader by searching for "people wearing masks", which resulted in a total of 100 images. The test images were preprocessed using Keras' ImageDataGenerator to rescale the pixel values and generate an image data generator. 

13. The previously loaded and compiled model was evaluated on the test dataset, achieving a test accuracy of 90% and a loss value of 0.6. To predict the class labels for the test images, the predict method of the model was used to obtain the probabilities, and binary class labels were determined based on a threshold of 0.5. The true and predicted labels were displayed for comparison. 
 
14. A function called prediction_plot was defined to visualize the images along with their true and predicted class labels, and the predictions for the first 20 test images were plotted.
