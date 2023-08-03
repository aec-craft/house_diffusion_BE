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


# for file_name in json_files:
#     if file_name.endswith(".json"):
#         file_path = os.path.join(data_folder, file_name)
#         with open(file_path, "r") as json_file:
#             file_data = json.load(json_file)
#             data["room_type"].append(file_data["room_type"])
#             data["boxes"].append(file_data["boxes"])
#             data["ed_rm"].append(file_data["ed_rm"])
#             data["edges"].append(file_data["edges"])
#
#
# def pad_sequences(sequences, padding_value):
#     max_length = max(len(seq) for seq in sequences)
#     padded_sequences = []
#
#     for seq in sequences:
#         if len(seq) < max_length:
#             padded_seq = seq + [padding_value] * (max_length - len(seq))
#         else:
#             padded_seq = seq[:max_length]  # Truncate longer sequences
#
#         padded_sequences.append(padded_seq)
#
#     return padded_sequences
#
#
# # Prepare the data
# room_type = np.array(pad_sequences(data["room_type"], 0), dtype=np.float32)
# box = np.array(pad_sequences(data["boxes"], [0, 0, 0, 0]), dtype=np.float32)
#
#
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(18, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 18 * 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = self.fc7(x)
        x = x.view(-1, 18, 4)
        return x


#
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.fc1 = nn.Linear(18 * 4, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 1)
#
#     def forward(self, x):
#         x = x.view(-1, 18 * 4)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.sigmoid(self.fc3(x))
#         return x
#
#
# # Create instances of generator and discriminator
# generator = Generator()
# discriminator = Discriminator()
#
# # Move models to device
# generator = generator.to(device)
# discriminator = discriminator.to(device)
#
# # Define optimizers
# generator_optimizer = optim.Adam(generator.parameters(), lr=0.01)
# discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.01)
#
# # Training loop
# num_epochs = 10000
# output_normalized = torch.from_numpy(box) / 255.0
# output_normalized = output_normalized.to(device)
# input_data = torch.from_numpy(room_type)
# input_data = input_data.to(device)
# for epoch in range(num_epochs):
#     # ---------------------
#     # Train discriminator
#     # ---------------------
#     discriminator_optimizer.zero_grad()
#
#     # Generate fake samples
#     fake_samples = generator(input_data)
#
#     # Calculate discriminator loss for real samples
#     real_output = discriminator(output_normalized)
#     real_labels = torch.ones_like(real_output)
#     real_loss = F.binary_cross_entropy(real_output, real_labels)
#
#     # Calculate discriminator loss for fake samples
#     fake_output = discriminator(fake_samples.detach())
#     fake_labels = torch.zeros_like(fake_output)
#     fake_loss = F.binary_cross_entropy(fake_output, fake_labels)
#
#     # Total discriminator loss
#     discriminator_loss = real_loss + fake_loss
#
#     # Backward pass and optimization for discriminator
#     discriminator_loss.backward()
#     discriminator_optimizer.step()
#
#     # ---------------------
#     # Train generator
#     # ---------------------
#     generator_optimizer.zero_grad()
#
#     # Generate fake samples
#     fake_samples = generator(input_data)
#
#     # Calculate generator loss
#     fake_output = discriminator(fake_samples)
#     generator_labels = torch.ones_like(fake_output)
#     generator_loss = F.binary_cross_entropy(fake_output, generator_labels)
#
#     # Backward pass and optimization for generator
#     generator_loss.backward()
#     generator_optimizer.step()
#
#     if (epoch + 1) % 500 == 0:
#         torch.save(generator.state_dict(), f"generator_{epoch + 1}.pth")
#         torch.save(discriminator.state_dict(), f"discriminator_{epoch + 1}.pth")
#
#     # Print the loss for monitoring the training progress
#     print(f"Epoch: {epoch + 1}/{num_epochs}, Generator Loss: {generator_loss}, Discriminator Loss: {discriminator_loss}")

generator = Generator()

generator.load_state_dict(torch.load("generator_1000.pth"))
generator.eval()
generator = generator.to(device)

# Generate room outputs
input_test = torch.from_numpy(np.array([1, 3, 4, 3, 17, 17, 17, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
input_test = input_test.to(device)
with torch.no_grad():
    fake_samples = generator(input_test)

# Convert the generated samples back to the original range
generated_outputs = fake_samples * 255.0

# Print the generated outputs
for i in range(generated_outputs.shape[0]):
    print(f"Generated Output {i + 1}:")
    print(generated_outputs[i])
