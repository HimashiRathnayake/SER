from utils.classifier import load_classifier 
from utils.data_handler import load_Sindhi_data, load_jl_corpus_with_random_testset, get_jl_corpus_statistics, load_jl_corpus_with_speaker_based_testset, load_RAVDESS_speech_corpus_with_random_testset, load_RAVDESS_speech_corpus_with_speaker_based_testset
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# get_jl_corpus_statistics()

X_train, X_test, y_train, y_test, emotion_categories = load_RAVDESS_speech_corpus_with_speaker_based_testset()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("[+] Number of training samples:", X_train.shape[0])
print("[+] Number of testing samples:", X_test.shape[0])
print("[+] Number of features:", X_train.shape[1])

classifier = "mlp" # lr: Logistic Regression Classifier | svm: Support Vector Machine | knn: KNeighbors Classifier | mlp: Multi-Layer Perceptron Classifier
model = load_classifier(classifier)
print("[*] Training the " + classifier + " model...")
model.fit(X_train, y_train.ravel())

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

# confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (6, 5))
cm = pd.DataFrame(confusion_matrix , index = [i for i in emotion_categories] , columns = [i for i in emotion_categories])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
font = {'size'   : 20}
plt.rc('font', **font)
plt.show()
