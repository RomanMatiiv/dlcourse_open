

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!

    tp = (prediction == 1) * (ground_truth == 1)
    fp = (prediction == 1) * (ground_truth == 0)
    tn = (prediction == 0) * (ground_truth == 0)
    fn = (prediction == 0) * (ground_truth == 1)

    tp = tp.astype(int).sum()
    fp = fp.astype(int).sum()
    tn = tn.astype(int).sum()
    fn = fn.astype(int).sum()

    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = 2 * (precision * recall)/(precision + recall)
    accuracy = (tp + tn)/(tp + tn + fp + fn)

    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''

    # TODO: Implement computing accuracy
    accuracy = (prediction == ground_truth).mean()

    return accuracy
