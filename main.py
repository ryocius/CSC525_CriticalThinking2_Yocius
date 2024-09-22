import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def takeInput(text, min = 0, max = 1000):
    try:
        inValue = float(input(text))
        if min <= inValue <= max:
            return inValue
        else:
            raise ValueError()
    except:
        print(f"Enter an float value between {min} and {max}")
        takeInput()


# Select the number of neighbors with the best accuracy value
def bestAccuracyKnn(XTrain, XTest, yTrain, yTest):
    accuracies = []
    for neighbor in range(1, 50):
        knn = KNeighborsClassifier(n_neighbors=neighbor)
        knn.fit(XTrain, yTrain)
        yPred = knn.predict(XTest)
        accuracy = accuracy_score(yTest, yPred)
        accuracies.append(accuracy)

    mostAccurateIndex = np.argmax(accuracies)
    mostAccurateNeighbor = mostAccurateIndex + 1
    maxAccuracy = accuracies[mostAccurateIndex]
    print(f"Best Value for N Neighbors = {mostAccurateNeighbor}")
    print(f'Accuracy: {maxAccuracy * 100:.2f}%')

    knn = KNeighborsClassifier(n_neighbors=mostAccurateNeighbor)
    knn.fit(XTrain, yTrain)

    return knn


def predictGenre(labelEncoder, knn):
    age = takeInput("Enter the person's age: ")
    height = takeInput("Enter the person's height in inches: ")
    weight = takeInput("Enter the person's weight in pounds: ")
    gender = takeInput("Enter the person's gender (0 for female and 1 for male): ", min=0, max=1)
    inputs = pd.DataFrame([[age, height, weight, gender]], columns=['age', 'height', 'weight', 'gender'])
    genreEncoded = knn.predict(inputs)[0]
    genre = labelEncoder.inverse_transform([genreEncoded])[0]
    if gender == 0:
        textGender = "Female"
    else:
        textGender = "Male"
    print(f"\nA {age} year old {textGender} who is {height} inches tall and {weight} "
          f"lbs most likely prefers the following video game genre: {genre}\n")


def main():
    data = pd.read_csv('data.csv')
    labelEncoder = LabelEncoder()
    data['genre'] = labelEncoder.fit_transform(data['genre'])

    X = data[['age', 'height', 'weight', 'gender']] # Features
    y = data['genre'] # Target

    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.20, random_state=56)

    knn = bestAccuracyKnn(XTrain, XTest, yTrain, yTest)

    while(True):
        predictGenre(labelEncoder, knn)


main()