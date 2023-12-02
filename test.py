from thundersvm import SVC
import torch
import pandas as pd

def prepare_data(data):
    train_x = data[data['Usage'] != 'PublicTest']
    test_x = data[data['Usage'] == 'PublicTest']
    train_y = train_x['emotion']
    test_y = test_x['emotion']
    train_x = train_x['pixels'].str.split(' ', expand=True).astype('float32')
    test_x = test_x['pixels'].str.split(' ', expand=True).astype('float32')
    return train_x, train_y, test_x, test_y

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
path = './data/fer2013/fer2013.csv'
data = pd.read_csv(path)
print("start loading data")
train_x, train_y, test_x, test_y = prepare_data(data)
print("loading data done")
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# Move data to GPU
# train_x, train_y, test_x, test_y = (
#     torch.tensor(train_x.values).to(device),
#     torch.tensor(train_y.values).to(device),
#     torch.tensor(test_x.values).to(device),
#     torch.tensor(test_y.values).to(device),
# )
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
# Create SVM model on GPU
face = SVC(
    C=1.0, kernel='sigmoid', gamma=0.01, cache_size=800,
    decision_function_shape='ovr', probability=True,
    random_state=42, gpu_id=5, verbose=1
)

# Train and test on GPU
face.fit(train_x, train_y)
print(face.score(test_x, test_y))