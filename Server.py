import flwr as fl
import numpy as np
from typing import List, Optional, Tuple

from Args import args


class SaveFedAvgModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
            self, 
            rnd: int, 
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException]
    ) -> Optional[fl.common.Weights]:
        
        weights = super().aggregate_fit(rnd, results, failures)
        if weights is not None:
            print(f"Saving round {rnd} weights...", flush=True)
            weights_file = f"weights/round-{rnd}-weights.npy"
            np.save(args.output_folder + weights_file, weights)
        
        return weights
