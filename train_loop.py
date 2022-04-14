from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.datautil.iterators import get_balanced_batches
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


rng = RandomState((2018,8,7))

# Print some statistics each epoch
def evaluate(model, data, config):
    model.eval()
    accuracies = []
    losses = []
    i_trials_in_batch = get_balanced_batches(len(data.X), rng, shuffle=True, batch_size=16)
    for i_trials in i_trials_in_batch:
        batch_X = np.array(data.X)[i_trials][:,:,:,None]
        batch_y = np.array(data.y)[i_trials]
        batch_X = np_to_var(batch_X)        
        if config.cuda:
            batch_X = batch_X.cuda()
        batch_y = np_to_var(batch_y)
        if config.cuda:
            batch_y = batch_y.cuda()
        outputs = model(batch_X)
        
        loss = F.nll_loss(outputs, batch_y)
        outputs = outputs.cpu().detach().numpy()
        batch_y = batch_y.cpu().detach().numpy()
        predicted_labels = np.argmax(outputs, axis=1)
        accuracy = accuracy_score(batch_y, predicted_labels)
        accuracies.append(accuracy)
        losses.append(loss.item())
    
    print('Accuracy: ', np.mean(accuracies))
    print('Loss: ', np.mean(losses))
    

def train(model, optimizer, train_set, valid_set, num_epochs):
    for i_epoch in range(1, num_epochs):
        print('Epoch ', i_epoch)
        i_trials_in_batch = get_balanced_batches(len(train_set.X), rng, shuffle=True,
                                            batch_size=16)
        # Set model to training mode
        model.train()
        for i_trials in tqdm(i_trials_in_batch):
        # Have to add empty fourth dimension to X
        
            batch_X = np.array(train_set.X)[i_trials][:,:,:,None]
            batch_y = np.array(train_set.y)[i_trials]
            batch_X = np_to_var(batch_X)
            if config.cuda:
                batch_X = batch_X.cuda()
            batch_y = np_to_var(batch_y)
            if config.cuda:
                batch_y = batch_y.cuda()
            # Remove gradients of last backward pass from all parameters
            optimizer.zero_grad()
            # Compute outputs of the network
            outputs = model(batch_X)
            # Compute the loss
            loss = F.nll_loss(outputs, batch_y)
            # Do the backpropagation
            loss.backward()
            # Update parameters with the optimizer
            optimizer.step()
        evaluate(model, train_set, config)
        evaluate(model, valid_set, config)
