## Face Mask Detection Using MobileNetV2 and YOLOv4

**Introduction:**

In the era of the COVID-19 pandemic, wearing a face mask has become essential to reduce the spread of the virus. Automated face mask detection systems can help enforce mask-wearing rules in public spaces. This project aims to develop a deep learning model for detecting whether a person is wearing a mask or not using images. The model uses MobileNetV2 as the base model for transfer learning and applies image data augmentation techniques to improve performance. Additionally, we implement the YOLO v4 object detection model for face mask detection on images.

**Objective:**

The project's main objective is to develop a deep learning-based face mask detection system that accurately identifies whether individuals are wearing masks, assisting in monitoring and ensuring compliance with public health regulations. To achieve this, in the first part, the project will utilize a dataset of images, employ MobileNetV2 with transfer learning techniques to create the face mask classifier, and implement a data augmentation pipeline to enhance the classifier's generalization capabilities. In the second part, YOLOv4 object detection model implemented to detect if a person is wearing mask or not in the image.

**MobileNetV2:**

MobileNetV2 is a lightweight and efficient deep learning model primarily designed for mobile and embedded vision applications. It is an improvement over the original MobileNet model, focusing on reducing computational complexity while maintaining high accuracy levels. MobileNetV2 uses depthwise separable convolutions, a technique that significantly reduces the number of parameters and computations required when compared to traditional convolutional layers. The key innovation in MobileNetV2 is the introduction of the inverted residual structure with linear bottlenecks. This architecture starts with a low-dimensional representation and expands it through a series of layers, followed by depthwise convolution and projection back to the lower-dimensional representation. This approach allows MobileNetV2 to retain more information through the network while minimizing computational costs. Furthermore, MobileNetV2 employs shortcut connections between bottlenecks, which aids in mitigating the vanishing gradient problem and improves overall performance.

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

**Second Part: Face Mask Detection**

1. The saved classifier model for mask detection was loaded and the YOLOv4 object detection model was downloaded from GitHub along with its configuration file and weights.

2. The downloaded configuration file and model weight are loaded using the OpenCV DNN module to identify faces in an image and then use the mask classifier model to determine whether the person is wearing a mask or not.

3.  A test image was loaded and preprocessed to be compatible with YOLOv4, followed by a forward pass through the network to obtain the detections. The output was filtered based on a confidence threshold, and non-maximum suppression was applied to remove overlapping bounding boxes.

4. The pre-trained mask classification model was then applied to the face regions within the bounding boxes. The face regions were extracted, resized, normalized, and prepared for input to the mask classifier.

5. The mask classifier made predictions for each face, determining whether the individual was wearing a mask or not. Based on the predictions, bounding boxes and labels were drawn on the original image, with different colors representing the presence or absence of a mask. Finally, the annotated image was displayed, showing the face mask detection results.

**Result and Analysis:**

After training the face mask classifier model using MobileNetV2 as a base and adding custom dense layers, the model achieved a test accuracy of 90%. This meant that the model could correctly classify 90% of the test images into their respective classes (with mask and without mask). The test loss was 0.60, indicating good performance as lower loss values suggested better model predictions. Overall, these results showed that the model performed well on the test set and was able to generalize to unseen data.

The predictions made on the test dataset, which contained 100 images downloaded from Bing, showed that the model could predict the presence or absence of face masks with reasonable accuracy. However, as the training data only contained single person with a face mask, the model struggled to classify images with more than one person. In such cases, the model couldn't accurately classify all individuals in the picture. Interestingly, the model could detect cartoon images with masks and worked well with meme images, demonstrating its versatility in handling different types of input images. When visualizing the predictions, it was evident that the model was generally successful in distinguishing between the two classes, though there were some cases where the predictions were incorrect. Some reasons for incorrect predictions could have been due to variations in image quality, lighting conditions, and occlusions.

In the second part of the project, a YOLO v4 object detection model was used to detect faces in images. Combining this with the face mask classifier model allowed for accurate detection of people wearing face masks in a given image. However, some errors still existed, particularly in detecting all the faces when there were many people in the pictures. This limitation might have been due to the complexity of the scenes, occlusions, or overlapping faces, which made it challenging for the model to identify and distinguish each individual accurately.

**Fine-tunning and Optimizations:**

By refining the model and its parameters, performance can be improved to handle various situations and complexities more efficiently. Here are some strategies to consider for fine-tuning and optimization:

1. **Data Augmentation:** We could enhance the training dataset by applying data augmentation techniques in the ImageDataGenerator for training data. Adjusting the parameters for rotation, flipping, scaling, and translation, will help the model generalize better to different image conditions and variations.

2. **Increase Training Data:** Gathering more diverse training data, including images with multiple people, various face mask styles, and different backgrounds, will improve the model's ability to handle complex images with multiple faces and different mask types.

3. **Optimize Model Architecture:** We could experiment with the architecture of the custom dense layers added to the MobileNetV2 base model. Also, can try adding more layers or changing the number of nodes in each layer, as well as exploring different activation functions and dropout rates to reduce overfitting.

4. **Adjust Training Parameters:** Fine-tuning the training parameters such as the learning rate, batch size, and the number of epochs may help improve the convergence and generalization of the model during training.

5. **Improve YOLO v4 Performance:** In the second part of the project, consider fine-tuning the YOLO v4 object detection model to better detect faces in images with multiple people by adjusting the configuration file and training the model on a dataset that contains images with multiple faces, or by using a pre-trained model specifically optimized for detecting faces.

6. **Ensemble Models:** Utilizing an ensemble of different models or model architectures to improve the overall performance of the face mask detection system. This could help to reduce the impact of errors made by one single model.

**Conclusion:**

In summary, this project developed a deep learning-based face mask detection system to enforce mask-wearing rules and support public health safety. Using MobileNetV2, transfer learning, and data augmentation, the model achieved 90% test accuracy, demonstrating strong performance on unseen data. However, the system faced challenges detecting faces in images with multiple people. To enhance performance, strategies such as increasing training data diversity, modifying model architecture, and fine-tuning the YOLOv4 object detection model can be employed. These improvements will contribute to a more robust and accurate face mask detection system.

**Citations:**

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Retrieved from https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_and_CVPR_2018_paper.html

2. Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv preprint arXiv:2004.10934. Retrieved from https://arxiv.org/abs/2004.10934
