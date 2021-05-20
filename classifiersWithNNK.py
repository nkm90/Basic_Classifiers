from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd


def neuralNetworkClassifier():
    # define the neural network model with keras
    model = Sequential()
    # Adding the layers
    model.add(Dense(12, input_dim=56,
                    activation='relu'))  # 1st layer with 12 neurons and 56 inputs
    model.add(Dense(8, activation='relu'))  # relu activation provides better performance
    model.add(Dense(1, activation='sigmoid'))  # using sigmoid activation I ensure outputs of 0 and 1
    # compile the keras model. Configures the model for training
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Trains the model for a fixed number of epochs and the number of samples per gradient update (change the weights)
    model.fit(X, Y,
              epochs=10,
              batch_size=2,
              verbose=0)
    return model


# load the data
data = pd.read_csv("dataset.csv")
print(data)
# select the different columns to specify which one are the x and the result y
X = data.values[:, 0:len(data.columns) - 2]
Y = data.values[:, len(data.columns) - 1]
# split the dataset and set the distance for the test side
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2)
print(Y)

# Initializing classifiers
clf1 = KNeighborsClassifier(n_neighbors=5)
clf2 = RandomForestClassifier(random_state=0)
clf3 = KerasClassifier(build_fn=neuralNetworkClassifier)

eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
                              weights=[1, 2, 3],
                              voting='soft')

for clf, label in zip([clf1, clf2, clf3, eclf],
                      ['KNN',
                       'Random Forest',
                       'Neural Network',
                       'Ensemble Vote']):
    # if label != 'Neural Network':

    while label != 'Neural Network':
        # train set calculation and in order
        clf = clf.fit(X_train, Y_train)
        # test the prediction
        Y_prediction = clf.predict(X_test)
        # compare train against test and print the accuracy
        print("Accuracy:", accuracy_score(Y_test, Y_prediction))
        print("Precision score:", precision_score(Y_test,
                                                  Y_prediction,
                                                  average="macro"))
        break
    # set number of splits
    splits = [3, 5, 10]
    for x in splits:
        nSplits = x
        # cross validation of the results provided by the test
        cv = ShuffleSplit(n_splits=nSplits, test_size=(1/nSplits), random_state=1)

        print(str(nSplits) + "-fold cross validation:")
        scores = model_selection.cross_val_score(clf, X, Y,
                                                 cv=cv,
                                                 scoring='accuracy')
        print("Accuracy: %0.4f (+/- %0.4f) [%s]"
              % (scores.mean(), scores.std(), label))

        precision = model_selection.cross_val_score(clf, X, Y,
                                                    cv=cv,
                                                    scoring='precision')
        print("Precision:  %0.4f (+/- %0.4f) [%s]"
              % (precision.mean(), precision.std(), label))

        auc = model_selection.cross_val_score(clf, X, Y,
                                              cv=cv,
                                              scoring='roc_auc')
        print("Roc_auc score:  %0.4f (+/- %0.4f) [%s]"
              % (auc.mean(), auc.std(), label))

        print("------------------------------------------------------------")
