from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.datautil.iterators import get_balanced_batches
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from models import build_model
from add_noise import addDataNoise
rng = RandomState((2018,8,7))


def evaluate(model, X, y, config, loss_func, eval=False, squeeze=False):
    """print metrics each epoch"""
    model.eval()
    accuracies = []
    losses = []
    i_trials_in_batch = get_balanced_batches(len(X), rng, 
                                             shuffle=True, 
                                             batch_size=config.batch_size)
    for i_trials in i_trials_in_batch:
        batch_X = X[i_trials][:,:,:,None]
        batch_y = np.array(y)[i_trials]
        batch_X = np_to_var(batch_X)   
        batch_y = np_to_var(batch_y)     
        if config.cuda:
            batch_X = batch_X.cuda()
            batch_y = batch_y.cuda()
        outputs = model(batch_X)

        if not squeeze:
            loss = loss_func(outputs, batch_y)
            outputs = outputs.cpu().detach().numpy()
            batch_y = batch_y.cpu().detach().numpy()
            predicted_labels = np.argmax(outputs, axis=1)
        else:
            outputs = outputs.squeeze().float()
            batch_y = batch_y.squeeze().float()
            loss = loss_func(outputs, batch_y)
            outputs = outputs.cpu().detach().numpy()
            batch_y = batch_y.cpu().detach().numpy()
            predicted_labels = [1 if i>=0.5 else 0 for i in outputs]
        
        accuracy = accuracy_score(batch_y, predicted_labels)
        accuracies.append(accuracy)
        losses.append(loss.item())
    
    print('Accuracy: ', np.mean(accuracies), ', Loss: ', np.mean(losses))
    return np.mean(accuracies)


def train(config, model, optimizer, trainX, trainy, 
          validX, validy, loss_func, squeeze=False):
    valid_accuracies = []
    train_accuracies = []
    for i_epoch in range(1, config.max_epochs):
        i_trials_in_batch = get_balanced_batches(len(trainX), rng, 
                                                 shuffle=True,
                                                 batch_size=config.batch_size)
        # Set model to training mode
        model.train()
        for i_trials in i_trials_in_batch:
        # Have to add empty fourth dimension to X
        
            batch_X = trainX[i_trials][:,:,:,None]
            batch_y = np.array(trainy)[i_trials]
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
            if not squeeze:
                pass
            else:
                outputs = outputs.squeeze().float()
                batch_y = batch_y.squeeze().float()
            # Compute the loss
            loss = loss_func(outputs, batch_y)
            # Do the backpropagation
            loss.backward()
            # Update parameters with the optimizer
            optimizer.step()
        print('Epoch ', i_epoch)
        print('========')
        print('Training Metrics: ')
        train_acc = evaluate(model, trainX, trainy, config, loss_func, squeeze=squeeze)
        print('Validation Metrics: ')
        val_acc = evaluate(model, validX, validy, config, loss_func, squeeze=squeeze)
        train_accuracies.append(train_acc)
        valid_accuracies.append(val_acc)
        
    return model, train_accuracies, valid_accuracies



def kf_training(X,y,test_X,test_y, model_name):
    models = []
    if model_name == 'deep':
        loss_func = F.nll_loss
        squeeze=False
    elif model_name == 'chronoNet':
        loss_func = nn.BCEWithLogitsLoss()
        squeeze=True
    train_accuracies = []
    valid_accuracies = []
    print('MODEL ', model_name, ':')
    for val_idx in range(0, config.n_folds):
        train_set, valid_set, test_set = split(config, X, y, test_X,test_y, val_idx)
        trainset_X, validset_X, testset_X = standardize(train_set, valid_set, test_set)
        model = build_model(config, model_name=model_name)
        optimizer = optim.Adam(model.parameters(), lr=config.init_lr)
        model, train_acc, val_acc = train(config, model, optimizer, trainset_X,
                                          train_set.y, validset_X, valid_set.y,
                                          loss_func=loss_func, squeeze=squeeze)
        train_accuracies.append(train_acc)
        valid_accuracies.append(val_acc)
        models.append(model)
        del valid_set, train_set, trainset_X, validset_X
    return models, train_accuracies, valid_accuracies, testset_X, test_set.y


def save_accuracies(models, testset_X, testset_y, band_cut=True):
    delta = [[1, 4]]
    theta = [[4, 8]]
    alpha = [[8, 12]]
    mu = [[12, 16]]
    beta = [[16, 25]]
    gamma = [[25, 40]]
    bands = [delta, theta, alpha, mu, beta, gamma]
    accuracies = {'no':[], 'delta': [], 'theta': [],
                  'alpha':[], 'mu': [], 'beta': [], 'gamma': []}
    band_names = list(accuracies.keys())[1:]
    for model in models:
        acc = evaluate(model, testset_X, testset_y,
                       config, nn.BCEWithLogitsLoss(),
                       squeeze=True, eval=True)

        accuracies['no'].append(acc)
        for band, band_name in zip(bands, band_names):
            test_noisy_band = addDataNoise(testset_X, band=band,
                                           srate=100, band_cut=band_cut)
            acc = evaluate(model, test_noisy_band, testset_y,
                           config, nn.BCEWithLogitsLoss(),
                           squeeze=True, eval=True)
            accuracies[band_name].append(acc)

    return accuracies
