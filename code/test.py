from glob import glob
import os
import torch
from animal_face_dataset import *
from torch.utils.data.sampler import SubsetRandomSampler
from Animal_Classification_Network import *



def main():
    #######################################################################
    # TODO: Extend the binary classification to multi-class classification
    N_CLASSES = 20 # num of classes
    #######################################################################



    # !!! DO NOT MAKE ANY CHANGES AFTER THIS LINE !!!



    # Load test dataset
    #########################################################################################################
    label_map = {0: "Cat", 1: "Dog", 2: "Bear", 3: "Chicken", 4: "Cow", 5: "Deer", 6: "Duck", 7: "Eagle",
                 8: "Elephant", 9: "Human", 10: "Lion", 11: "Monkey", 12: "Mouse", 13: "Panda", 14: "Pigeon",
                 15: "Pig", 16: "Rabbit", 17: "Sheep", 18: "Tiger", 19: "Wolf"}
    main_path = r"C:\Users\alaah\Desktop\AER1515-Perception_for_Robotics\AER1515_Assignment1\AER1515_Assignment1\AnimalFace\test"
    paths = []
    labels = []

    for i in range(N_CLASSES):
        folder = label_map[i] + 'Head'
        path_i = os.path.join(main_path, folder, "*")
        for each_file in glob(path_i):
            paths.append(each_file)
            labels.append(i)
    dataset = AnimalDataset(paths, labels, (150, 150))

    dataset_indices = list(range(0, len(dataset)))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    print("Number of test samples: ", len(dataset_indices))
    #########################################################################################################



    # Set up device (gpu or cpu), load CNN model
    ######################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(N_CLASSES).to(device)
    # Load your saved model here!
    model.load_state_dict(torch.load("best_model.pt"))
    ######################################################################



    # Start the test
    ######################################################
    total_true = 0
    total = len(dataset_indices)
    with torch.no_grad():
        model.eval()
        for data_, target_ in test_loader:
            data_ = data_.to(device)
            target_ = target_.to(device)

            outputs = model(data_)
            _, preds = torch.max(outputs, dim=1)
            true = torch.sum(preds == target_).item()
            total_true += true

    test_accuracy = round(100 * total_true / total, 2)
    print(f"Test accuracy: {test_accuracy}%")
    ######################################################

    return

if __name__ == '__main__':
    main()