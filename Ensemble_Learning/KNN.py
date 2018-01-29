class KNN(MLModel):
    def train(self, x_train, y_train):
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(self.x_train, self.y_train)

    def report_accuracy(self, x_text, y_test):
        self.knn.score(self.x_test, self.y_test)
