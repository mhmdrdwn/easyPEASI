def kruskal_test(acc_dict):
    accuracies = []
    bands = []
    for band, acc in acc_dict.items():
        accuracies.append(acc)
        bands.append(band)
    accuracies = np.array(accuracies)
    print(accuracies.shape)
    from scipy import stats
    for i in range(1, 7):
        print(bands[i], ":", stats.kruskal(accuracies[0, :], accuracies[i, :]))

