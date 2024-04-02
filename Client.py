import flwr as fl
import torch
from collections import OrderedDict
from datetime import datetime

from MyUtils import test, test_with_fpr
from ReinforcementUtils import reinforcement_train, dueling_reinforcement_train
from Args import args

def client_logic(net, train_loaders, test_loader, metrics):
        
    class CifarClient(fl.client.NumPyClient):
        
        def __init__(self):
            super().__init__()
            self.split_id = 0
            self.weight_multiplier = 1
        
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]
        
        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            
        def fit(self, parameters, config):
            init_time = datetime.now()
            self.set_parameters(parameters)
            train_loader = train_loaders[self.split_id]
            
            print('Training on data on split id: ' + str(self.split_id), flush=True)
            self.split_id += 1
            
            dueling_reinforcement_train(net, train_loader)
            num_examples = len(train_loader.dataset)
            print('Samples in the current round: ' + str(num_examples), flush=True)
            print("Num examples in fit: " + str(num_examples))
            
            finish_time = datetime.now()
            print("Time taken for this fit round: ", (finish_time-init_time))
            return self.get_parameters(), int(num_examples*self.weight_multiplier), {}
            # return self.get_parameters(), num_examples, {}
        
        def evaluate(self, parameters, config):
            init_time = datetime.now()
            self.set_parameters(parameters)
            loss, accuracy, fpr = test_with_fpr(net, test_loader)
            num_examples = len(test_loader.dataset)
            
            print('Current weight multiplier: ' + str(self.weight_multiplier), flush=True)
            print("Num examples in eval: " + str(num_examples))
            print('Loss: ' + str(loss), flush=True)
            print('Accuracy: ' + str(accuracy), flush=True)
            print('FPR: ' + str(fpr), flush=True)
            
            metrics['accuracy'].append(accuracy)
            metrics['loss'].append(loss)
            metrics['attention_value'].append(self.weight_multiplier)
            
            cur_weight_multiplier = self.weight_multiplier
            self.weight_multiplier = self.fpr_based_weight_modifer(fpr)
            
            finish_time = datetime.now()
            print("Time taken for this evaluation round: ", (finish_time-init_time))
            print('', flush=True)
            return float(loss), int(num_examples*cur_weight_multiplier), {"accuracy": float(accuracy)}
            # return float(loss), num_examples, {"accuracy": float(accuracy)}
        
        def accuracy_based_weight_modifer(self, accuracy):
            return function_1(accuracy)
        def fpr_based_weight_modifer(self, fpr):
            return function_1(fpr)

            
        
    def start():
        fl.client.start_numpy_client("localhost:8080", client=CifarClient())
    
    
    return start


def function_1(x):
    k = args.fparam_k
    a = args.fparam_a
    return 1 + k*(1-x)*pow(a, -x)
    
# Exports: client_logic