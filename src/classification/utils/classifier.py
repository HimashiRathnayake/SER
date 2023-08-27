from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def load_classifier(classifier_name):

    if classifier_name == "lr": # Logistic Regression Classifier
        model_params = {
            'max_iter': 10000,
            'penalty' : 'l2',
        }
        model = LogisticRegression(**model_params)

    elif classifier_name == "svm": # Support Vector Machine
        model_params = {
            'kernel': 'rbf'
        }
        model = SVC(**model_params)

    elif classifier_name == "knn": # KNeighbors Classifier
        model = KNeighborsClassifier() 

    elif classifier_name == "mlp": # Multi-Layer Perceptron Classifier
        model_params = {
            'alpha': 0.01,
            'batch_size': 64,
            'epsilon': 1e-08, 
            'hidden_layer_sizes': (300,), 
            'learning_rate': 'adaptive', 
            'max_iter': 500, 
        }
        model = MLPClassifier(**model_params)

    return model
