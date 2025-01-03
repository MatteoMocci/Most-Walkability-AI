# Most Walkability AI Project
This GitHub Repository contains AI models for predicting walkability scores of roads.
A first model takes as input a picture from google maps' street view representing a road and returns a walkability score from 0 (worst score) to 4 (best score).
A second model takes satellite pictures of the same roads and returns a walkability score
A third and final model takes as input the features of the first and second model and returns a walkability score basically from the informations obtained from both the street-view and satellite pictures
The main.py executes code in order to train the three models and calculate the walkability for the data given

# Models Description
## ViT classifier for street view and satellite images
The first and second model are finetuned version of this ViT classifier from the [Hugging Face Hub](https://huggingface.co/google/vit-base-patch16-224).
The first model is finetuned on photos obtained from Street-View of points in the middle of roads in Cagliari, Sassari e Alghero
The second model is finetuned on photos of the same points but obtained through a 3D Terrain in Cesium for Unreal
In the Street-View Dataset each photo is labeled with the walkability score that the model has to return.
This label is also applied to the satellite pictures.

### Training Hyper-parameters for Model 1 and 2
The hyperparameters for training models 1 and 2 are:
- 10 epochs
- 1e-4 learning rate
- adamw_hf as optimizer
- batch_size = 16
- bf16 = True
- tf32 = True

### Metrics
The metrics that are used for evaluation are accuracy, recall, precision, mse, confusion matrix and a custom metric called one_out. The one_out_accuracy uses
the confusion matrix to check how many predictions of the model are within 1 from the ground truth (so label 2 is considered correct if ground truth is 1 or 3, incorrect if 0 or 5).
Since each label is actually a walkability score, this metric is useful to see how many predictions of the model are correct or pretty close to the expected value, and, thus,
how many predictions are way off (for example a street with 0 walkability score is predicted as 4)

### Results on Validation Set for Model 1
- Loss         : 0.6629
- Accuracy     : 76.8 %
- MSE          : 0.298
- Precision    : 79.1 %
- Recall       : 71.3 %
- Confusion matrix :
![Matrix](matrix/m1.png)
- One-off accuracy : 98.2 %

### Results on Test Set for Model 1
- Loss         :      0.6627
- Accuracy     :      73.7 %
- MSE          :      0.336
- Precision    :      69.8 %
- Recall       :      65.5%
- Confusion Matrix :
![Matrix](matrix/m2.png)
- One-off accuracy :  97.9 %

### Results on Validation Set for Model 2
- Loss         : 1.837
- Accuracy     : 70 %
- MSE          : 0.429
- Precision    : 68.9 %
- Recall       : 63.9 %
- Confusion matrix :
![Matrix](matrix/m3.png)
- One-off accuracy : 96.2 %

### Results on Test Set for Model 2
- Loss         :      0.7381
- Accuracy     :      69.7 %
- MSE          :      0.435
- Precision    :      66.5 %
- Recall       :      59.5%
- Confusion Matrix :
![Matrix](matrix/m4.png)
- One-off accuracy :  96.6 %

## SVC model
After the first two ViT models have been trained, their embeddings for training, validation and test are extracted. A feature selection is performed fitting a Linear SVC with penalty='l2' on the training features.
The features extracted from the two models are concatenated in a single array of features which is the input of the SVC model. The y to predict are the same walkability scores of the original instances, but this time we are starting from the most relevant informations extracted from
satellite and street-view pictures.

### Results on Validation Set for Model 3 
- Loss         : 0.474
- Accuracy     : 83 %
- MSE          : 0.191
- Precision    : 85 %
- Recall       : 79 %
- Confusion matrix :
![Matrix](matrix/m5.png)
- One-off accuracy : 99.3 %

### Results on Test Set for Model 3
- Loss         :      0.512
- Accuracy     :      83 %
- MSE          :      0.191
- Precision    :      82 %
- Recall       :      79 %
- Confusion Matrix :
![Matrix](matrix/m6.png)
- One-off accuracy :  99.6 %


