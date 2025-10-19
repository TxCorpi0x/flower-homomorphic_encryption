from server import *
import os
from core.benchmark import save_benchmark, print_benchmark

"""
Script to start the server side of the federated learning pipeline with Flower.
"""

if __name__ == "__main__":
    print("start server")
    start_time = time.time()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    print(f"Server Time = {time.time() - start_time} seconds")

    # Save server-side benchmark if enabled
    if args.benchmark:
        # Derive results directory from model_save or default
        if args.model_save:
            out_dir = os.path.dirname(args.model_save)
            os.makedirs(out_dir, exist_ok=True)
            bench_path = os.path.join(out_dir, "benchmark.json")
        else:
            bench_path = "./benchmark.json"
        save_benchmark(bench_path)
        print_benchmark()
