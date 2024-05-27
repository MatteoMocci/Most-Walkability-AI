# Most Walkability AI Project
This GitHub Repository contains an AI model for predicting walkability scores of roads.
This model takes as input a picture from google maps' street view representing a road and returns a walkability score from 0 (worst score) to 4 (best score).

# Training Hyper-parameters
This version's hyper-parameters for training are:
- 10 epochs
- 1e-4 learning rate
- adamw_hf as optimizer
- batch_size = 32
- fp16 = True

# Metrics
The metrics that are used for evaluation are accuracy, recall, precision, mse, confusion matrix and a custom metric called one_out. The one_out_accuracy uses
the confusion matrix to check how many predictions of the model are within 1 from the ground truth (so label 2 is considered correct if ground truth is 1 or 3, incorrect if 0 or 5).
Since each label is actually a walkability score, this metric is useful to see how many predictions of the model are correct or pretty close to the expected value, and, thus,
how many predictions are way off (for example a street with 0 walkability score is predicted as 4)

# Results on Test Set
- Loss         : 0.678
- Accuracy     : 72 %
- MSE          : 0.3334
- Precision    : 68 %
- Recall       : 66.9 %
- One-off accuracy : 98.2 %
