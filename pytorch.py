import torch
import torch.nn as nn
import os
import json
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_folder = "datasets/rplan"
json_files = os.listdir(data_folder)
data = {
    "room_type": [],
    "boxes": [],
    "edges": [],
    "ed_rm": []
}

output_min = 0.0
output_max = 255.0

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
room_type = np.array(pad_sequences(data["room_type"], 999), dtype=np.float32)
box = np.array(pad_sequences(data["boxes"], [9999, 9999, 9999, 9999]), dtype=np.float32)
edges = np.array(pad_sequences(data["edges"], [9999, 9999, 9999, 9999, 999, 999]), dtype=np.float32)

edrm=[]

ed_rm = []
for x in data["ed_rm"]:
    modified_data = [sublist + [999] if len(sublist) == 1 else sublist for sublist in x]
    ed_rm.append(modified_data)

ed_rm = np.array(pad_sequences(ed_rm, [999, 999]), dtype=np.float32)
#
# # Define the sequential model


class SequentialModel(nn.Module):
    def __init__(self):
        super(SequentialModel, self).__init__()

        # Define the layers for the first output
        self.fc1 = nn.Linear(18, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 18 * 4)  # Adjust the output shape

        # Define the layers for the second output
        self.fc8 = nn.Linear(18, 64)
        self.fc9 = nn.Linear(64, 128)
        self.fc10 = nn.Linear(128, 256)
        self.fc11 = nn.Linear(256, 128)
        self.fc12 = nn.Linear(128, 64)
        self.fc13 = nn.Linear(64, 32)
        self.fc14 = nn.Linear(32, 128 * 6)  # Adjust the output shape

        # Define the layers for the third output
        self.fc15 = nn.Linear(18, 64)
        self.fc16 = nn.Linear(64, 128)
        self.fc17 = nn.Linear(128, 256)
        self.fc18 = nn.Linear(256, 128)
        self.fc19 = nn.Linear(128, 64)
        self.fc20 = nn.Linear(64, 32)
        self.fc21 = nn.Linear(32, 128 * 2)  # Adjust the output shape

    def forward(self, x):
        # Pass the input through the layers for the first output
        x1 = torch.relu(self.fc1(x))
        x1 = torch.relu(self.fc2(x1))
        x1 = torch.relu(self.fc3(x1))
        x1 = torch.relu(self.fc4(x1))
        x1 = torch.relu(self.fc5(x1))
        x1 = torch.relu(self.fc6(x1))
        output1 = self.fc7(x1)
        output1 = output1.view(-1, 18, 4)

        # Pass the input through the layers for the second output
        x2 = torch.relu(self.fc8(x))
        x2 = torch.relu(self.fc9(x2))
        x2 = torch.relu(self.fc10(x2))
        x2 = torch.relu(self.fc11(x2))
        x2 = torch.relu(self.fc12(x2))
        x2 = torch.relu(self.fc13(x2))
        output2 = self.fc14(x2)
        output2 = output2.view(-1, 128, 6)

        # Pass the input through the layers for the third output
        x3 = torch.relu(self.fc15(x))
        x3 = torch.relu(self.fc16(x3))
        x3 = torch.relu(self.fc17(x3))
        x3 = torch.relu(self.fc18(x3))
        x3 = torch.relu(self.fc19(x3))
        x3 = torch.relu(self.fc20(x3))
        output3 = self.fc21(x3)
        output3 = output3.view(-1, 128, 2)

        return output1, output2, output3


# Create an instance of the sequential model
# model = SequentialModel()
#
# loss_function = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# # Training loop
# num_epochs = 10000
# output_normalized = torch.from_numpy(box)
# output_normalized2 = torch.from_numpy(edges)
# output_normalized3 = torch.from_numpy(ed_rm)
# model = model.to(device)
# input_data = torch.from_numpy(room_type)
# input_data = input_data.to(device)
# output_normalized = output_normalized.to(device)
# output_normalized2 = output_normalized2.to(device)
# output_normalized3 = output_normalized3.to(device)
# for epoch in range(num_epochs):
#     # Forward pass
#     output1, output2, output3 = model(input_data)
#
#     # Compute the loss
#     loss1 = loss_function(output1, output_normalized)
#     loss2 = loss_function(output2, output_normalized2)
#     loss3 = loss_function(output3, output_normalized3)
#
#     loss = loss1 + loss2 + loss3
#
#     # Backward pass and optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 500 == 0:
#         torch.save(model.state_dict(), f"model_{epoch + 1}.pth")
#     # Print the loss for monitoring the training progress
#     print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {loss}")

model = SequentialModel()
model_path = "model_2000.pth"
model.load_state_dict(torch.load(model_path))
input_test = torch.from_numpy(np.array([1, 3, 4, 3, 17, 17, 17, 15, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
                                       dtype=np.float32))  # Example input tensor for testing
output_1, output_2, output_3 = model(input_test)
print("Output:")
