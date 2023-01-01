import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import *
from datetime import timedelta
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, criterion, optimizer, data, l2 =3, batch=50, device = 'cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.L2 = max(l2, 0)
        train, train_label, test, test_label = data['train'].to(device), data['train_label'].to(device), data['test'].to(device), data['test_label'].to(device)
        self.trainLoader =  DataLoader(list(zip(train, train_label)), batch_size=batch, shuffle=True)
        self.testLoader = DataLoader(list(zip(test, test_label)), batch_size=batch, shuffle=False)
        if 'dev' in data:
            self.dev = True
            self.devLoader = DataLoader(list(zip(data['dev'].to(device), data['dev_label'].to(device))), batch_size=batch, shuffle=False)
            self.dev_list = []
        else:
            self.dev = False

        self.loss_list = []
        

    def train(self, epoch_size, show_batches = 1, patience=7, verbose=True):
        if verbose:
            start = time()
            timeiter = tqdm(total = len(self.trainLoader) * epoch_size, position=0, leave=True)
        if self.dev:
            dev_score = 0
            max_score = 0
            count=patience
        # Set Train Mode
        self.model.train()
        torch.enable_grad()

        for epoch in range(epoch_size):  # loop over the dataset multiple times

            running_loss = 0.0
            cut = 0
            for data in self.trainLoader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                #print(labels)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.L2)
                self.optimizer.step()

                running_loss += loss.item()
                cut += 1
                (lambda x: timeiter.update(1) if x else 0)(verbose)
                if cut % show_batches == 0:
                    current_loss = running_loss / show_batches
                    self.loss_list.append(current_loss)
                    running_loss = 0.0
                    cut = 0

                    if  verbose:
                    # print statistics        
                        timeiter.set_description(f"loss : {current_loss: .3f}")

            if self.dev: ## Early Stopping if Dev set exist
                dev_score = self.test(dev=True, verbose=False)
                self.dev_list.append(dev_score)

                if dev_score < max_score:
                    count -= 1
                    if count == 0:
                        print(f"Early Stopped at {epoch}_th epoch")
                        break
                else:
                    max_score = dev_score
                    count = patience
                # Re-Set for Training
                self.model.train()
                torch.enable_grad()
                
        
        if verbose:
            timeiter.close() 
            print(f"Finished Training : {str(timedelta(seconds=int(time() - start), ))} spent.")
        return
            
    def test(self, dev = False, verbose=True):
        correct = 0
        total = 0
        if dev:
            if self.dev:
                Loader = self.devLoader
            else:
                if verbose:
                    print("Dataset has No DEV set, so use TEST set")
                    Loader = self.testLoader
                    dev = False
        else:
            Loader = self.testLoader
        state = 'dev' if dev else 'test'
        
        # since we're not training, we don't need to calculate the gradients for our outputs
        self.model.eval()
        with torch.no_grad():
            for data in Loader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.model.forward(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if verbose:
            print(f'Accuracy of the Model on the {len(self.testLoader.dataset)} {state} set: {100 * correct / total:.2f} %') 
        return correct/total
    
    def plot(self):
        if len(self.loss_list)==0:
            raise ValueError("Train Model First")
        

        if self.dev:
            fig, ax_loss = plt.subplots()
            ax_loss.plot(self.loss_list, label='y1', color='green')
            ax_loss.set_ylabel('Loss')

            ax_right = ax_loss.twinx()
            ax_right.plot([dev*100 for dev in self.dev_list], label='Dev', color='orange')
            ax_right.set_ylabel('score(%)')
        
        else:
            fig = plt.plot(self.loss_list)

        plt.show(fig)
            