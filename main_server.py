from server import *
import os
from core.benchmark import init_benchmark, save_benchmark, print_benchmark

"""
Script to start the server side of the federated learning pipeline with Flower.
"""

if __name__ == "__main__":
    print("start server")
    # Initialize benchmarking if enabled
    if args.benchmark:
        mode = (
            "he" if args.he else ("zkp" if getattr(args, "zkp", False) else "baseline")
        )
        try:
            init_benchmark(mode, args.number_clients, args.rounds)
        except Exception:
            # Fallback with generic defaults
            init_benchmark(mode, 1, args.rounds)
    start_time = time.time()

    server_address = os.environ.get("FL_SERVER_ADDRESS", "0.0.0.0:8080")
    fl.server.start_server(
        server_address=server_address,
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
