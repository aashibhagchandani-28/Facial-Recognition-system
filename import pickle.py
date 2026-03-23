import pickle

with open("classifier.pkl", "rb") as f:
    model, class_names = pickle.load(f)

print(class_names)