import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import os
import json
import tensorflow as tf

# Dataset
data_folder = "datasets/rplan"
json_files = os.listdir(data_folder)
data = {
    "room_type": [],
    "boxes": [],
    "edges": [],
    "ed_rm": []
}


for file_name in json_files:
    if file_name.endswith(".json"):
        file_path = os.path.join(data_folder, file_name)
        with open(file_path, "r") as json_file:
            file_data = json.load(json_file)
            data["room_type"].append(file_data["room_type"])
            data["boxes"].append(file_data["boxes"])
            data["ed_rm"].append(file_data["ed_rm"])
            data["edges"].append(file_data["edges"])


def pad_sequences(sequences, padding_value):
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []

    for seq in sequences:
        if len(seq) < max_length:
            padded_seq = seq + [padding_value] * (max_length - len(seq))
        else:
            padded_seq = seq[:max_length]  # Truncate longer sequences

        padded_sequences.append(padded_seq)

    return padded_sequences


# Prepare the data
room_type = pad_sequences(data["room_type"], 0)
box = pad_sequences(data["boxes"], [0, 0, 0, 0])
room_types = np.array(room_type)
boxes = np.array(box)

input_shape = (18,)

# Define the output shape
output_shape = (18, 4)
# Define the model
model = Sequential()
model.add(keras.layers.InputLayer(input_shape=input_shape))

# Add any desired layers
model.add(keras.layers.Reshape(output_shape))

# Compile the model
model.compile(loss="mse", optimizer="adam")
# Train the model
model.fit(room_types, boxes, epochs=100, batch_size=32, validation_split=0.8)
# Make predictions
room_types_test = np.array([1, 2, 3, 17, 17, 15])  # Example test input
predictions = model.predict(room_types_test)
print("Predicted Boxes:")
for i, prediction in enumerate(predictions):
    print(f"Room Type {room_types_test[i]}: {prediction}")