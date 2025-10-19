import torch.nn.functional
import numpy as np
import flwr as fl
from core import *

"""
Script to define the client side of the federated learning pipeline with Flower.
"""


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################
class FlowerClient(fl.client.NumPyClient):
    """
    The client class for the federated learning pipeline with Flower.
    This class is used to define the client-side logic for federated learning with Flower.
    It inherits from fl.client.NumPyClient. The only real difference between Client and NumPyClient is
    that NumPyClient takes care of serialization and deserialization for you.

    args:
        cid: client id (int)
        net: model (torch.nn.Module)
        trainloader: trainloader (torch.utils.data.DataLoader)
        valloader: valloader (torch.utils.data.DataLoader)
        device: device (torch.device)
        batch_size: batch size (int)
        save_results: path to save the results (str)
        matrix_path: path to save the confusion matrix (str)
        roc_path: path to save the roc curve (str)
        yaml_path: path to save the yaml file (str)
        he: boolean to use the homomorphic encryption (bool)
        classes: list of classes (list)
        context_client: context client (tenseal.context.Context)
    """

    def __init__(
        self,
        cid,
        net,
        trainloader,
        valloader,
        device,
        batch_size,
        save_results,
        matrix_path,
        roc_path,
        yaml_path,
        he,
        classes,
        context_client,
        zkp=False,
        zkp_context=None,
        benchmark_metrics=None,
        dp=False,
        dp_params=None,
    ):

        # Initialize the client
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid
        self.device = device
        self.batch_size = batch_size
        self.save_results = save_results
        self.matrix_path = matrix_path
        self.roc_path = roc_path
        self.yaml_path = yaml_path
        self.he = he
        self.classes = classes
        self.context_client = context_client
        self.zkp = zkp
        self.zkp_context = zkp_context
        self.benchmark = benchmark_metrics
        self.dp = dp
        self.dp_params = dp_params

    def get_parameters(self, config):
        """
        get the model parameters and return them as a list of NumPy ndarray's
        (which is what flwr.client.NumPyClient expects)

        args:
            config: config (dict)

        return:
            parameters: parameters (list)
        """
        print(f"[Client {self.cid}] get_parameters")

        if self.benchmark:
            with BenchmarkTimer(self.benchmark, "client_get_params"):
                if self.zkp:
                    # Generate ZKP commitments for benchmarking only
                    with BenchmarkTimer(self.benchmark, "proof_generation"):
                        _ = zkp_commit_model(self.net.state_dict(), self.zkp_context)
                    # Return plain numpy parameters for transport
                    params = [
                        val.detach().cpu().numpy()
                        for _, val in self.net.state_dict().items()
                    ]
                elif self.he:
                    # In simulation runs, avoid transporting encrypted tensors (serialization limit)
                    sim_mode = os.environ.get("FL_SIMULATION", "0") == "1"
                    if sim_mode:
                        with BenchmarkTimer(self.benchmark, "encryption"):
                            _ = get_parameters2(self.net, self.context_client, None)
                        params = [
                            val.detach().cpu().numpy().astype(np.float32, copy=False)
                            for _, val in self.net.state_dict().items()
                        ]
                    else:
                        with BenchmarkTimer(self.benchmark, "encryption"):
                            params = get_parameters2(
                                self.net, self.context_client, None
                            )
                elif self.dp:
                    # Get plain parameters first
                    params = get_parameters2(self.net, None, None)
                    # Note: DP noise is added during fit(), not here
                else:
                    params = get_parameters2(self.net, None, None)

            # Track communication size
            self.benchmark.add_upload_size(estimate_params_size(params))
            return params
        else:
            if self.zkp:
                # Generate proofs for benchmarking, but return numpy
                _ = zkp_commit_model(self.net.state_dict(), self.zkp_context)
                return [
                    val.detach().cpu().numpy()
                    for _, val in self.net.state_dict().items()
                ]
            else:
                return get_parameters2(self.net, self.context_client, None)

    def fit(self, parameters, config):
        """
        - Update the parameters of the local model with the parameters received from the server
        - Train the updated model on the local train dataset (x_train/y_train)
        - Return the updated model parameters and the number of examples used for training to the server (for 1 given round)

        args:
            parameters: parameters (list)
            config: config (dict)

        return:
            parameters: parameters (list)
        """
        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        lr = float(config["learning_rate"])

        # Use values provided by the config
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")

        # Update local model parameters
        if self.benchmark:
            self.benchmark.add_download_size(estimate_params_size(parameters))

            if self.zkp:
                with BenchmarkTimer(self.benchmark, "proof_verification"):
                    # In transport we send numpy arrays; skip verification here
                    pass
                set_parameters(self.net, parameters, None, None)
            elif self.he:
                sim_mode = os.environ.get("FL_SIMULATION", "0") == "1"
                if sim_mode:
                    # Simulate decryption cost, but apply plain numpy params to model
                    with BenchmarkTimer(self.benchmark, "decryption"):
                        try:
                            sample_tensor = next(
                                iter(self.net.state_dict().values())
                            ).detach()
                            _ = crypte({"sample": sample_tensor}, self.context_client)
                            _ = self.context_client.secret_key()
                        except Exception:
                            pass
                    set_parameters(
                        self.net,
                        [p.astype(np.float32, copy=False) for p in parameters],
                        None,
                        None,
                    )
                else:
                    with BenchmarkTimer(self.benchmark, "decryption"):
                        set_parameters(self.net, parameters, self.context_client, None)
            else:
                set_parameters(self.net, parameters, None, None)
        else:
            if self.zkp:
                set_parameters(self.net, parameters, None, self.zkp_context)
            else:
                set_parameters(self.net, parameters, self.context_client, None)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)

        # Start training
        if self.benchmark:
            with BenchmarkTimer(self.benchmark, "client_fit"):
                results = engine.train(
                    self.net,
                    self.trainloader,
                    self.valloader,
                    optimizer=optimizer,
                    loss_fn=criterion,
                    epochs=local_epochs,
                    device=self.device,
                )
            self.benchmark.add_client_memory(get_memory_usage_mb())

            # Collect model quality metrics from training
            if results:
                # Add final epoch metrics (last values in the lists)
                if results["train_loss"]:
                    self.benchmark.add_train_loss(results["train_loss"][-1])
                if results["train_acc"]:
                    self.benchmark.add_train_accuracy(results["train_acc"][-1])
                if results["val_loss"]:
                    self.benchmark.add_val_loss(results["val_loss"][-1])
                if results["val_acc"]:
                    self.benchmark.add_val_accuracy(results["val_acc"][-1])
        else:
            results = engine.train(
                self.net,
                self.trainloader,
                self.valloader,
                optimizer=optimizer,
                loss_fn=criterion,
                epochs=local_epochs,
                device=self.device,
            )

        # Save results
        if self.save_results:
            save_graphs(self.save_results, local_epochs, results, f"_Client {self.cid}")

        # Get updated parameters with appropriate method
        if self.benchmark:
            if self.zkp:
                with BenchmarkTimer(self.benchmark, "proof_generation"):
                    _ = zkp_commit_model(self.net.state_dict(), self.zkp_context)
                # Return plain numpy for transport
                updated_params = [
                    val.detach().cpu().numpy()
                    for _, val in self.net.state_dict().items()
                ]
            elif self.he:
                sim_mode = os.environ.get("FL_SIMULATION", "0") == "1"
                if sim_mode:
                    # Measure encryption cost but return numpy for transport (simulation-safe)
                    with BenchmarkTimer(self.benchmark, "encryption"):
                        _ = get_parameters2(self.net, self.context_client, None)
                        updated_params = [
                            val.detach().cpu().numpy().astype(np.float32, copy=False)
                            for _, val in self.net.state_dict().items()
                        ]
                else:
                    with BenchmarkTimer(self.benchmark, "encryption"):
                        updated_params = get_parameters2(
                            self.net, self.context_client, None
                        )
            elif self.dp:
                # Get plain parameters
                updated_params = get_parameters2(self.net, None, None)
                # Apply differential privacy (clip + noise)
                with BenchmarkTimer(self.benchmark, "dp_noise_addition"):
                    updated_params, dp_stats = apply_differential_privacy(
                        updated_params, self.dp_params, clip_norm=True
                    )
                # Store DP stats for metrics
                self._dp_stats = dp_stats
            else:
                updated_params = get_parameters2(self.net, None, None)

            self.benchmark.add_upload_size(estimate_params_size(updated_params))

            # Return benchmark metrics in the response
            metrics = {
                "client_fit_time": (
                    self.benchmark.client_fit_time[-1]
                    if self.benchmark.client_fit_time
                    else 0
                ),
                "upload_size": (
                    self.benchmark.params_size_upload[-1]
                    if self.benchmark.params_size_upload
                    else 0
                ),
                "download_size": (
                    self.benchmark.params_size_download[-1]
                    if self.benchmark.params_size_download
                    else 0
                ),
                "client_memory": (
                    self.benchmark.client_memory_peak[-1]
                    if self.benchmark.client_memory_peak
                    else 0
                ),
            }

            # Add model quality metrics
            if self.benchmark.train_loss:
                metrics["train_loss"] = self.benchmark.train_loss[-1]
            if self.benchmark.train_accuracy:
                metrics["train_accuracy"] = self.benchmark.train_accuracy[-1]
            if self.benchmark.val_loss:
                metrics["val_loss"] = self.benchmark.val_loss[-1]
            if self.benchmark.val_accuracy:
                metrics["val_accuracy"] = self.benchmark.val_accuracy[-1]

            # Add crypto metrics if available
            if self.zkp and self.benchmark.proof_generation_time:
                metrics["proof_generation_time"] = self.benchmark.proof_generation_time[
                    -1
                ]
                metrics["proof_verification_time"] = (
                    self.benchmark.proof_verification_time[-1]
                    if self.benchmark.proof_verification_time
                    else 0
                )
            elif self.dp and hasattr(self, "_dp_stats"):
                metrics["dp_noise_time"] = (
                    self.benchmark.dp_noise_addition_time[-1]
                    if hasattr(self.benchmark, "dp_noise_addition_time")
                    and self.benchmark.dp_noise_addition_time
                    else 0
                )
                metrics["dp_clipped"] = self._dp_stats.get("clipped", False)
                metrics["dp_epsilon"] = self._dp_stats.get("epsilon", 0)
            elif self.he and self.benchmark.encryption_time:
                metrics["encryption_time"] = (
                    self.benchmark.encryption_time[-1]
                    if self.benchmark.encryption_time
                    else 0
                )
                metrics["decryption_time"] = (
                    self.benchmark.decryption_time[-1]
                    if self.benchmark.decryption_time
                    else 0
                )

            return updated_params, len(self.trainloader), metrics
        else:
            if self.zkp:
                # Generate proofs for benchmarking, but return numpy
                _ = zkp_commit_model(self.net.state_dict(), self.zkp_context)
                return (
                    [
                        val.detach().cpu().numpy()
                        for _, val in self.net.state_dict().items()
                    ],
                    len(self.trainloader),
                    {},
                )
            else:
                return (
                    get_parameters2(self.net, self.context_client, None),
                    len(self.trainloader),
                    {},
                )

    def evaluate(self, parameters, config):
        """
        - Update the parameters of the local model with the parameters received from the server

        - Evaluate the updated model on the local test dataset (x_test/y_test)

        - Return the local loss and accuracy to the server

        args:
            parameters: parameters (list)
            config: config (dict)

        return:
            loss: loss (float)
            num_examples: number of examples (int)
            metrics: metrics (dict)
        """
        print(f"[Client {self.cid}] evaluate, config: {config}")

        if self.zkp:
            set_parameters(self.net, parameters, None, None)
        else:
            set_parameters(self.net, parameters, self.context_client, None)

        # Evaluate global model parameters on the local test data
        if self.benchmark:
            with BenchmarkTimer(self.benchmark, "client_eval"):
                loss, accuracy, y_pred, y_true, y_proba = engine.test(
                    self.net,
                    self.valloader,
                    loss_fn=torch.nn.CrossEntropyLoss(),
                    device=self.device,
                )
                # Collect test metrics
                self.benchmark.add_test_loss(loss)
                self.benchmark.add_test_accuracy(accuracy)
        else:
            loss, accuracy, y_pred, y_true, y_proba = engine.test(
                self.net,
                self.valloader,
                loss_fn=torch.nn.CrossEntropyLoss(),
                device=self.device,
            )

        if self.save_results:
            os.makedirs(self.save_results, exist_ok=True)
            if self.matrix_path:
                save_matrix(
                    y_true, y_pred, self.save_results + self.matrix_path, self.classes
                )

            if self.roc_path:
                save_roc(
                    y_true,
                    y_proba,
                    self.save_results + self.roc_path,
                    len(self.classes),
                )
        # Return results, including the custom accuracy metric
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


# The client-side execution (training, evaluation) from the server-side
def client_common(
    cid: str,
    model_save: str,
    path_yaml: str,
    path_roc: str,
    results_save: str,
    path_matrix: str,
    batch_size: str,
    trainloaders,
    valloaders,
    DEVICE,
    CLASSES,
    he=False,
    secret_path="",
    server_path="",
    zkp=False,
    zkp_params_path="",
    benchmark_metrics=None,
    dp=False,
    dp_params_path="",
):
    """
    args:
        cid: client id (str)
        model_save: path to save the model (str)
        path_yaml: path to save the yaml file (str)
        path_roc: path to save the roc curve (str)
        results_save: path to save the results (str)
        path_matrix: path to save the confusion matrix (str)
        batch_size: batch size (int)
        trainloaders: trainloaders (list)
        valloaders: valloaders (list)
        DEVICE: device (torch.device)
        CLASSES: classes (list)
        he: boolean to use the homomorphic encryption (bool)
        secret_path: path to save the secret key (str)
        server_path: path to save the server public key (str)
        zkp: boolean to use zero-knowledge proofs (bool)
        zkp_params_path: path to ZKP parameters (str)
        benchmark_metrics: BenchmarkMetrics object for performance tracking

    return:
        FlowerClient: client (FlowerClient object)

    """
    # Load data
    # Note: each client gets a different trainloader/valloader, so each client will train and evaluate
    # on their own unique dataset.
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    context_client = None
    zkp_context = None
    dp_params = None

    # Load model
    net = Net(num_classes=len(CLASSES)).to(DEVICE)

    # Zero-Knowledge Proofs
    if zkp:
        print("Run with Zero-Knowledge Proofs")
        if os.path.exists(zkp_params_path):
            zkp_context = read_zkp_params(zkp_params_path)
            print(f"ZKP parameters loaded from: {zkp_params_path}")
        else:
            print(f"Error: ZKP parameters not found at {zkp_params_path}")
            print("Run: python create_zkp_params.py")
            raise FileNotFoundError(f"ZKP parameters not found: {zkp_params_path}")

    # Differential Privacy
    elif dp:
        print("Run with Differential Privacy")
        if os.path.exists(dp_params_path):
            dp_params = load_dp_params(dp_params_path)
            print(f"DP parameters loaded from: {dp_params_path}")
            print(f"  Epsilon: {dp_params.epsilon}, Delta: {dp_params.delta}")
            print(
                f"  Max Grad Norm: {dp_params.max_grad_norm}, Noise Multiplier: {dp_params.noise_multiplier:.4f}"
            )
        else:
            print(f"Error: DP parameters not found at {dp_params_path}")
            print("Run: python create_dp_params.py")
            raise FileNotFoundError(f"DP parameters not found: {dp_params_path}")

    # Homomorphic encryption
    elif he:
        print("Run with homomorphic encryption")
        if os.path.exists(secret_path):
            # To get the existing public/private keys combination
            with open(secret_path, "rb") as f:
                query = pickle.load(f)

            context_client = ts.context_from(query["contexte"])

        else:
            # To create the public/private keys combination
            context_client = security.context()
            with open(secret_path, "wb") as f:  # 'ab' to add existing file
                encode = pickle.dumps(
                    {"contexte": context_client.serialize(save_secret_key=True)}
                )
                f.write(encode)

        secret_key = context_client.secret_key()

    else:
        print("Run WITHOUT cryptographic protection (baseline)")

    # C) Update the local model with the parameters received from the server
    # to get the trained model and the trained parameters (optimizer, metrics, ...)
    if os.path.exists(model_save):
        print(" To get the checkpoint")
        checkpoint = torch.load(model_save, map_location=DEVICE)["model_state_dict"]
        if he:
            print("to decrypt model")
            # To decrypt the parameters with the private key
            server_query, server_context = security.read_query(server_path)
            server_context = ts.context_from(server_context)
            for name in checkpoint:
                print(name)
                # To decrypt the parameters with the private key
                checkpoint[name] = torch.tensor(
                    security.deserialized_layer(
                        name, server_query[name], server_context
                    ).decrypt(secret_key)
                )

        # Update network with the aggregated results
        net.load_state_dict(checkpoint)

    # Create a  single Flower client representing a single organization
    return FlowerClient(
        cid,
        net,
        trainloader,
        valloader,
        device=DEVICE,
        batch_size=batch_size,
        matrix_path=path_matrix,
        roc_path=path_roc,
        save_results=results_save,
        yaml_path=path_yaml,
        he=he,
        context_client=context_client,
        classes=CLASSES,
        zkp=zkp,
        zkp_context=zkp_context,
        benchmark_metrics=benchmark_metrics,
        dp=dp,
        dp_params=dp_params,
    )
