import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score


def evaluate(model, dataset):
    print(f"Number of items: {dataset.base_num_items}")
    print(f"Number of choices: {len(dataset)}")
    print(f"Number of parameters: {sum(w.shape[1] for w in model.trainable_weights)}")
    print("Average Negative Log-Likelihood:", model.evaluate(dataset).numpy())
    print(
        "Total Negative Log-Likelihood:", model.evaluate(dataset).numpy() * len(dataset)
    )

    probas = model.predict_probas(dataset)
    print("Min probability: ", np.min(probas))
    print("Max probability: ", np.max(probas))

    predicted_choices, actual_choices = np.argmax(probas, axis=1), dataset.choices

    accuracy = accuracy_score(actual_choices, predicted_choices)
    f1 = f1_score(actual_choices, predicted_choices, average="macro")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
