import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def takeInput(text, min = 0, max = 1000):
    try:
        inValue = int(input(text))
        if min < inValue < max:
            return inValue
        else:
            raise ValueError()
    except:
        print(f"Enter an integer value between {min} and {max}")
        takeInput()


def predictGenre():
    age = takeInput("Enter the person's age: ")
    height = takeInput("Enter the person's height in inches: ")
    weight = takeInput("Enter the person's weight in pounds: ")
    gender = takeInput("Enter the person's gender (0 for female and 1 for male): ", min = 0, max = 1)


def main():
    data = pd.read_csv('your_data.csv')