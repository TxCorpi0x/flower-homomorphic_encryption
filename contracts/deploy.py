"""
Smart Contract Deployment Script for Federated Learning
"""

import json
import os
from web3 import Web3
from solcx import compile_source, install_solc


def compile_contract(contract_path: str) -> dict:
    """Compile the Solidity contract"""

    # Install solc if not present
    try:
        install_solc("0.8.19")
    except:
        print("Solc already installed or installation failed")

    # Read contract source
    with open(contract_path, "r") as file:
        contract_source = file.read()

    # Compile contract
    compiled_sol = compile_source(contract_source, output_values=["abi", "bin"])

    # Get contract interface
    contract_id, contract_interface = compiled_sol.popitem()

    return contract_interface


def deploy_contract(
    w3: Web3,
    contract_interface: dict,
    account: str,
    private_key: str,
    min_clients: int = 2,
    max_duration: int = 3600,
) -> tuple:
    """Deploy the contract to the blockchain"""

    # Get contract
    contract = w3.eth.contract(
        abi=contract_interface["abi"], bytecode=contract_interface["bin"]
    )

    # Build constructor transaction
    constructor_txn = contract.constructor(min_clients, max_duration).build_transaction(
        {
            "from": account,
            "gas": 3000000,
            "gasPrice": w3.to_wei("20", "gwei"),
            "nonce": w3.eth.get_transaction_count(account),
        }
    )

    # Sign transaction
    signed_txn = w3.eth.account.sign_transaction(constructor_txn, private_key)

    # Send transaction
    tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)

    # Wait for transaction receipt
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    return tx_receipt, contract_interface["abi"]


def save_deployment_info(contract_address: str, abi: list, network: str = "localhost"):
    """Save deployment information for later use"""

    deployment_info = {
        "network": network,
        "contract_address": contract_address,
        "abi": abi,
        "deployment_time": int(time.time()),
    }

    # Create deployments directory
    os.makedirs("./deployments", exist_ok=True)

    # Save deployment info
    with open(f"./deployments/{network}_deployment.json", "w") as f:
        json.dump(deployment_info, f, indent=2)

    print(f"Deployment info saved to ./deployments/{network}_deployment.json")


def main():
    """Main deployment function"""
    import time

    # Configuration
    CONTRACT_PATH = "./contracts/FederatedLearning.sol"
    RPC_URL = "http://localhost:8545"  # Local blockchain
    PRIVATE_KEY = "0x" + "0" * 63 + "1"  # Replace with actual private key
    MIN_CLIENTS_PER_ROUND = 2
    MAX_ROUND_DURATION = 3600  # 1 hour

    # Connect to blockchain
    w3 = Web3(Web3.HTTPProvider(RPC_URL))

    if not w3.is_connected():
        print("Failed to connect to blockchain!")
        print("Make sure you have a local blockchain running (e.g., Ganache, Hardhat)")
        return

    # Get account
    account = w3.eth.account.from_key(PRIVATE_KEY)
    account_address = account.address

    print(f"Connected to blockchain: {RPC_URL}")
    print(f"Deploying from account: {account_address}")
    print(f"Account balance: {w3.eth.get_balance(account_address) / 10**18} ETH")

    # Compile contract
    print("Compiling contract...")
    try:
        contract_interface = compile_contract(CONTRACT_PATH)
        print("✓ Contract compiled successfully")
    except Exception as e:
        print(f"✗ Contract compilation failed: {e}")
        return

    # Deploy contract
    print("Deploying contract...")
    try:
        tx_receipt, abi = deploy_contract(
            w3,
            contract_interface,
            account_address,
            PRIVATE_KEY,
            MIN_CLIENTS_PER_ROUND,
            MAX_ROUND_DURATION,
        )

        contract_address = tx_receipt.contractAddress
        print(f"✓ Contract deployed successfully!")
        print(f"Contract address: {contract_address}")
        print(f"Transaction hash: {tx_receipt.transactionHash.hex()}")
        print(f"Gas used: {tx_receipt.gasUsed}")

    except Exception as e:
        print(f"✗ Contract deployment failed: {e}")
        return

    # Save deployment info
    save_deployment_info(contract_address, abi, "localhost")

    # Test basic functionality
    print("\nTesting contract functionality...")
    try:
        contract = w3.eth.contract(address=contract_address, abi=abi)

        # Get current round
        current_round = contract.functions.currentRound().call()
        print(f"Current round: {current_round}")

        # Get min clients per round
        min_clients = contract.functions.minClientsPerRound().call()
        print(f"Min clients per round: {min_clients}")

        print("✓ Contract is working correctly!")

    except Exception as e:
        print(f"✗ Contract test failed: {e}")

    print(f"\nDeployment completed!")
    print(f"Contract address: {contract_address}")
    print(f"You can now use this address in your FL configuration:")
    print(f"  --contract_address {contract_address}")


if __name__ == "__main__":
    main()
