import numpy as np
import json
import os

from sklearn.model_selection import train_test_split


def load_data(dirname: str) -> list[np.array]:
    data = []
    files = [os.path.join(dirname, file) for file in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, file))]

    print("Loading", len(files), "files as data")
    for file in files:
        try:
            with open(file) as f:
                content = f.read()
        except:
            print("Error in: ", file)
            continue

        content = content.replace("(", "[").replace(")", "]")

        for line in content.split('\n'):
            if ('[' in line) and (']' in line):
                try:
                    arr = json.loads(line)
                    if isinstance(arr, list):
                        img = np.array(arr)
                        if img.size != 101 or ((img[1:] == -1) | (img[1:] == 1)).sum() != 100:
                            print("Error in: ", file)
                        else:
                            data.append(img)
                except:
                    print("Error in: ", file)

    print(len(data), " images loaded as data")

    return data


def prepare_data(data, letter1, letter2, algorithm):
    # Filter the data for the given letters
    filtered_data = [x for x in data if x[0] in [letter1, letter2]]
    if not filtered_data:
        print(f"No data found for labels {letter1} and {letter2}")
        return None, None, None, None
    # Split the data into features and labels
    features, labels = zip(*[(item[1:], item[0]) for item in filtered_data])
    features = np.array(features)
    labels = np.array(labels)

    # Convert the labels of letter1 to 1 and letter2 to 0 (or -1)

    if algorithm == "adaline":
        labels = np.where(labels == letter1, 1, -1)
    else:
        labels = np.where(labels == letter1, 1, 0 if letter2 else -1)

    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
