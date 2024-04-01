class Args:
    def __init__(self):
        self.epochs = 100
        self.lr = 0.01
        self.random_state = 42
        self.test_set_size = 0.1
        self.batch_size = 64
        self.per_exponent = 2
        self.output_folder = 'script_outputs/'
        self.output_file_suffix = '-output.txt'
        self.metrics_file_suffix = '-metrics.npy'
        
        self.nsl_columns = 33
        self.isot_columns = 85
        
        self.agent_data_splits = 200
        self.num_clients = 10
        self.fparam_k = 30
        self.fparam_a = 50
        
# =============================================================================
#         Training Specifications
# =============================================================================
#         self.dataset = 'isot'
        self.dataset = 'nsl'
        self.data_split_type = 'random'
#         self.data_split_type = 'customized'
# =============================================================================
        
        if(self.dataset == 'nsl'):
            self.n_columns = self.nsl_columns            
        else:
            self.n_columns = self.isot_columns
            
        
        if(self.data_split_type == 'customized'):
            if(self.dataset == 'nsl'):
                self.num_clients = 2
                self.fparam_k = 50000
                self.fparam_a = 200
            else:
                self.num_clients = 5
        
        
args = Args()

# Exports: args