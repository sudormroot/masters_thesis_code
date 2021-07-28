from sklearn.neighbors import KNeighborsClassifier

def knn_accuracy(X_train, y_train, X_test, y_test, k = 1):
    
    knn = KNeighborsClassifier(n_neighbors = k)
    
    knn.fit(X_train, y_train)
    
    accuracy_train = knn.score(X_train, y_train)
    accuracy_test = knn.score(X_test, y_test)
    
    return accuracy_train, accuracy_test
