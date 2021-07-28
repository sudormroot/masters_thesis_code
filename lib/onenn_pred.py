from sklearn.neighbors import KNeighborsClassifier

def onenn_accuracy(X_train, y_train, X_test, y_test):
    
    knn = KNeighborsClassifier(n_neighbors = 1)
    
    knn.fit(X_train, y_train)
    
    acc = knn.score(X_test, y_test)
    
    print("accuracy on testing dataset: ", acc)

    return acc
