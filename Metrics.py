class Metrics:
    
    def __init__(self, y_actual, y_hat):
        self.y_actual = y_actual
        self.y_hat = y_hat
        
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.accuracy = -1
        self.fpr = -1
        self.tpr = -1
        self.recall = -1
        self.precision = -1
        self.f1_score = -1
        
        self.compute_all_metrics()
        
    def compute_all_metrics(self):
    
        for i in range(len(self.y_hat)): 
            if (self.y_actual[i] == 1) and (self.y_hat[i] == 1):
                self.TP += 1
            if (self.y_hat[i] == 1) and (self.y_actual[i] != self.y_hat[i]):
                self.FP += 1
            if (self.y_actual[i] == 0) and (self.y_hat[i] == 0):
                self.TN += 1
            if (self.y_hat[i] == 0) and (self.y_actual[i] != self.y_hat[i]):
                self.FN += 1
        
        acc = self.compute_accuracy(self.TP, self.FP, self.TN, self.FN)
        fpr = self.compute_fpr(self.TP, self.FP, self.TN, self.FN)
        recall = self.compute_tpr(self.TP, self.FP, self.TN, self.FN)
        prec = self.compute_precision(self.TP, self.FP, self.TN, self.FN)
        
        self.f1_score = (2*prec*recall)/(prec+recall)
        
    
    def get_all_metrics(self):
        return self.accuracy, self.fpr, self.recall, self.precision
        
    
    def compute_accuracy(self, TP, FP, TN, FN):
        self.accuracy = (TP+TN)/(TP+TN+FP+FN)
        return self.accuracy
    
    def compute_fpr(self, TP, FP, TN, FN):
        self.fpr = (FP)/(FP+TN)
        return self.fpr
    
    def compute_tpr(self, TP, FP, TN, FN):
        self.tpr = (TP)/(TP+FN)
        self.recall = self.tpr
        return self.tpr
    
    def compute_precision(self, TP, FP, TN, FN):
        self.precision = (TP)/(TP+FP)
        return self.precision