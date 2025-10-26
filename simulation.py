from client import *
from server import *

"""
Script to start the simulation of the federated learning pipeline with Flower (client and server).
"""

# Global benchmark metrics collector
global_benchmark_metrics = None


def aggregate_fit_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Aggregate fit metrics from clients and collect benchmark data."""
    global global_benchmark_metrics
    # Extract benchmark metrics if available
    if global_benchmark_metrics is not None:
        for num_examples, client_metrics in metrics:
            if "client_fit_time" in client_metrics:
                global_benchmark_metrics.add_client_fit(
                    client_metrics["client_fit_time"]
                )
            if "upload_size" in client_metrics:
                global_benchmark_metrics.add_upload_size(client_metrics["upload_size"])
            if "download_size" in client_metrics:
                global_benchmark_metrics.add_download_size(
                    client_metrics["download_size"]
                )
            if "client_memory" in client_metrics:
                global_benchmark_metrics.add_client_memory(
                    client_metrics["client_memory"]
                )

            # Model quality metrics
            if "train_loss" in client_metrics:
                global_benchmark_metrics.add_train_loss(client_metrics["train_loss"])
            if "train_accuracy" in client_metrics:
                global_benchmark_metrics.add_train_accuracy(
                    client_metrics["train_accuracy"]
                )
            if "val_loss" in client_metrics:
                global_benchmark_metrics.add_val_loss(client_metrics["val_loss"])
            if "val_accuracy" in client_metrics:
                global_benchmark_metrics.add_val_accuracy(
                    client_metrics["val_accuracy"]
                )

            # Crypto-specific metrics
            if "proof_generation_time" in client_metrics:
                global_benchmark_metrics.add_proof_generation(
                    client_metrics["proof_generation_time"]
                )
            if "proof_verification_time" in client_metrics:
                global_benchmark_metrics.add_proof_verification(
                    client_metrics["proof_verification_time"]
                )
            if "encryption_time" in client_metrics:
                global_benchmark_metrics.add_encryption(
                    client_metrics["encryption_time"]
                )
            if "decryption_time" in client_metrics:
                global_benchmark_metrics.add_decryption(
                    client_metrics["decryption_time"]
                )
            if "dp_noise_time" in client_metrics:
                global_benchmark_metrics.add_dp_noise(client_metrics["dp_noise_time"])

    # Return empty dict as we're just collecting, not aggregating for strategy
    return {}


def aggregate_evaluate_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Aggregate evaluation metrics and collect global model quality data."""
    global global_benchmark_metrics

    # Collect test/evaluation metrics from clients if available
    if global_benchmark_metrics is not None:
        for num_examples, client_metrics in metrics:
            if "accuracy" in client_metrics:
                # This is from client evaluate()
                pass  # Already collected via client_eval in client.py

    # Compute weighted average for accuracy (standard Flower aggregation)
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    if sum(examples) > 0:
        aggregated_accuracy = sum(accuracies) / sum(examples)
        return {"accuracy": aggregated_accuracy}

    return {}


def evaluate_with_benchmark(server_round: int, parameters, config):
    """Wrapper around evaluate2 to capture global model quality metrics."""
    global global_benchmark_metrics

    # Call the original evaluate2 function
    result = evaluate2(server_round, parameters, config)

    # Capture metrics if benchmarking is enabled
    if result is not None and global_benchmark_metrics is not None:
        loss, metrics = result
        global_benchmark_metrics.add_global_val_loss(loss)
        if "accuracy" in metrics:
            global_benchmark_metrics.add_global_val_accuracy(metrics["accuracy"])

    return result


# Implementing a Flower client
def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    return client_common(
        cid,
        model_save,
        path_yaml,
        path_roc,
        results_save,
        path_matrix,
        batch_size,
        trainloaders,
        valloaders,
        DEVICE,
        CLASSES,
        he,
        he_backend,
        secret_path,
        server_path,
        zkp,
        zkp_params_path,
        benchmark_metrics,
        dp,
        dp_params_path,
    )


# ////////////////////////////// Simulation of the federated learning pipeline with Flower ////////////////////////////
# Pass parameters to the Strategy for server-side parameter initialization
if __name__ == "__main__":
    # Mark simulation mode for client logic that must avoid non-serializable payloads
    os.environ["FL_SIMULATION"] = "1"
    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    """
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}
    """
    model_save = args.model_save
    path_yaml = args.yaml_path
    path_roc = args.roc_path
    results_save = args.save_results
    path_matrix = args.matrix_path
    batch_size = args.batch_size
    he = args.he
    he_backend = args.he_backend if hasattr(args, "he_backend") else "tenseal"
    zkp = args.zkp
    dp = args.dp
    secret_path = args.path_keys
    server_path = args.path_crypted
    zkp_params_path = args.zkp_params
    dp_params_path = args.dp_params

    # Initialize benchmarking if enabled
    benchmark_metrics = None
    if args.benchmark:
        mode = "he" if he else ("zkp" if zkp else ("dp" if dp else "baseline"))
        benchmark_metrics = init_benchmark(mode, args.number_clients, args.rounds)
        global_benchmark_metrics = benchmark_metrics
        print(f"Benchmarking enabled for mode: {mode}")

    print("Start simulation")

    # Create strategy with benchmark metrics aggregation if enabled
    if args.benchmark:
        # Import strategy creation from server
        custom_strategy = FedCustom(
            fraction_fit=args.frac_fit,
            fraction_evaluate=args.frac_eval,
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=(
                args.min_eval_clients
                if args.min_eval_clients
                else args.number_clients // 2
            ),
            min_available_clients=args.min_avail_clients,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,  # Add benchmark collection
            initial_parameters=ndarrays_to_parameters(get_parameters2(central)),
            evaluate_fn=(None if args.he else evaluate_with_benchmark),
            on_fit_config_fn=get_on_fit_config_fn(
                epoch=args.max_epochs, lr=args.lr, batch_size=args.batch_size
            ),
            context_client=server_context,
        )
    else:
        custom_strategy = strategy  # Use the default strategy from server.py

    start_simulation = time.time()
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.number_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=custom_strategy,
        # client_resources=client_resources
    )
    simulation_time = time.time() - start_simulation
    print(f"Simulation Time = {simulation_time} seconds")

    # Save benchmark results if enabled
    if args.benchmark and benchmark_metrics:
        if results_save:
            benchmark_path = f"{results_save}/benchmark.json"
        else:
            benchmark_path = "./benchmark.json"
        benchmark_metrics.save(benchmark_path)
        benchmark_metrics.print_summary()
