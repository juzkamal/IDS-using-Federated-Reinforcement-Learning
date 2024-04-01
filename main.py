import flwr as fl
import numpy as np
import time
from multiprocessing import Process
import sys
from datetime import datetime

from Net import Net
from Server import SaveFedAvgModelStrategy
from Client import client_logic
from MyUtils import load_data, delete_files
from Args import args

from Data import get_nsl_random_splits
from Data import get_isot_random_splits
from Data import get_nsl_customized_splits
from Data import get_isot_customized_splits


splits = []

if(args.dataset == 'nsl'):
    if(args.data_split_type == 'random'):
        splits = get_nsl_random_splits()
    else:
        splits = get_nsl_customized_splits()
else:
    if(args.data_split_type == 'random'):
        splits = get_isot_random_splits()
    else:
        splits = get_isot_customized_splits()


def start_server():
    sys.stdout = open(args.output_folder + 'server' + args.output_file_suffix, 'w')
    
    print('Datatset: ' + args.dataset)
    print('Split type: ' + args.data_split_type)
    print('')
    
    # Define strategy
    save_fedAvg_strategy = SaveFedAvgModelStrategy(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_available_clients = args.num_clients,
        min_fit_clients = args.num_clients,
        min_eval_clients = args.num_clients
    )
    
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": args.agent_data_splits},
        strategy=save_fedAvg_strategy
    )
    
    sys.stdout.close()
    

def start_client(client_id):
    
    file_name = 'client-' + str(client_id) + args.output_file_suffix
    sys.stdout = open(args.output_folder + file_name, 'w')
    
    x, y = splits[client_id]
    train_loaders, test_loader = load_data(x, y)
    
    net = Net()
    metrics = {"accuracy" : [], "loss" : [], "attention_value":[]}
    
    start_fn = client_logic(net, train_loaders, test_loader, metrics)
    start_fn()
    
    sys.stdout.close()
    metrics_file = 'client-' + str(client_id) + args.metrics_file_suffix
    np.save(args.output_folder + metrics_file, metrics)
    
    
def main():
    
    # This will hold all the processes which we are going to create
    processes = []
    
    # Start the server
    server_process = Process(target=start_server)
    server_process.start()
    processes.append(server_process)
    
    # Blocking the script here for few seconds, so the server has time to start
    time.sleep(5)
    
    # Start all the clients
    for i in range(args.num_clients):
        client_process = Process(target=start_client, args=(i,))
        client_process.start()
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()
        


if __name__ == "__main__":
    init_time = datetime.now()
    delete_files(args.output_folder + '*' + args.output_file_suffix)
    delete_files(args.output_folder + '*' + args.metrics_file_suffix)
    delete_files(args.output_folder + 'weights/*.npy')

    print('Datatset: ' + args.dataset)
    print('Split type: ' + args.data_split_type)

    main()
    fin_time = datetime.now()
    print("Total execution time: ", (fin_time-init_time))