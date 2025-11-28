import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

model = None
dataset = None

print("Average Negative Log-Likelihood:", model.evaluate(dataset).numpy())
print("Total Negative Log-Likelihood:", model.evaluate(dataset).numpy() * len(dataset))

probas = model.predict_probas(dataset)

predicted_choices, actual_choices = np.argmax(probas, axis=1), dataset.choices

accuracy = accuracy_score(actual_choices, predicted_choices)
f1 = f1_score(actual_choices, predicted_choices, average="macro")

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")


def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


predicted_utilities = model.compute_batch_utility(
    shared_features_by_choice=dataset.shared_features_by_choice,
    items_features_by_choice=dataset.items_features_by_choice,
    available_items_by_choice=dataset.available_items_by_choice,
    choices=None,
)

final_utilities = np.mean(predicted_utilities, axis=0)
final_utilities = normalize(final_utilities) * 10

predicted_probabilities = np.round(np.mean(probas, axis=0) * 100, 3)

print("Min probability:", np.min(predicted_probabilities))
print("Max probability:", np.max(predicted_probabilities))

report = pd.DataFrame(
    {"utility": final_utilities, "probability": predicted_probabilities}
)
