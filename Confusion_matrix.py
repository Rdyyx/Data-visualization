import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def data_confusion_matrix(label, pred, model_name):
    conf_matrix = confusion_matrix(label, pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d",
                xticklabels=digits.target_names, yticklabels=digits.target_names)
    plt.title(f"Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    plt.savefig('confusion_matrix.jpg')


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_digits
    digits = load_digits()#这里使用手写数字数据集
    img, label = digits.data, digits.target
    img_train, img_test, label_train, label_test = train_test_split(img, label, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(img_train, label_train)
    pred = model.predict(img_test)

    model_name = 'model'
    data_confusion_matrix(label_test, pred, model_name)


