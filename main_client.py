from client import *
from federated import *

"""
script to start the client with the federated learning pipeline with Flower.
"""

if __name__ == "__main__":
    print("start client")

    # Initialize benchmarking if enabled
    benchmark_metrics = None
    if args.benchmark:
        mode = "he" if args.he else ("zkp" if args.zkp else "baseline")
        # Client doesn't know rounds count in non-simulation mode, use default
        rounds = getattr(args, "rounds", 1)
        benchmark_metrics = init_benchmark(mode, 1, rounds)  # Single client

    start_time = time.time()
    server_address = os.environ.get("FL_SERVER_ADDRESS", "[::]:8080")
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client_common(
            args.id_client,
            args.model_save,
            args.yaml_path,
            args.roc_path,
            args.save_results,
            args.matrix_path,
            args.batch_size,
            trainloaders,
            valloaders,
            DEVICE,
            CLASSES,
            args.he,
            args.he_backend if hasattr(args, "he_backend") else "tenseal",
            args.path_keys,
            args.path_crypted,
            args.zkp,
            args.zkp_params,
            benchmark_metrics,
        ),
    )
    client_time = time.time() - start_time
    print(f"client Time = {client_time} seconds")

    # Save benchmark if enabled
    if args.benchmark and benchmark_metrics:
        benchmark_path = (
            f"{args.save_results}/client_{args.id_client}_benchmark.json"
            if args.save_results
            else f"./client_{args.id_client}_benchmark.json"
        )
        benchmark_metrics.save(benchmark_path)
        benchmark_metrics.print_summary()
