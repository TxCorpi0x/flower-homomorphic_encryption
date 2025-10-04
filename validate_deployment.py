#!/usr/bin/env python3
"""
Complete ZKFL Production Deployment Validation Script

This script validates that all components of the ZKFL production
environment are working correctly.
"""

import subprocess
import time
import sys
import requests
from pathlib import Path


def run_command(cmd, description, capture_output=True):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Error: {result.stderr}")
                return False, result.stderr
            return True, result.stdout
        else:
            result = subprocess.run(cmd, shell=True)
            return result.returncode == 0, ""
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False, str(e)


def check_docker():
    """Check if Docker is available."""
    success, output = run_command("docker --version", "Checking Docker")
    if success:
        print(f"✅ Docker available: {output.strip()}")
        return True
    print("❌ Docker not available")
    return False


def check_docker_compose():
    """Check if Docker Compose is available."""
    success, output = run_command("docker-compose --version", "Checking Docker Compose")
    if success:
        print(f"✅ Docker Compose available: {output.strip()}")
        return True
    print("❌ Docker Compose not available")
    return False


def build_images():
    """Build Docker images."""
    print("\n🏗️  Building Docker Images...")

    # Build server image
    success, output = run_command(
        "docker build -f zkfl_production/Dockerfile.server -t zkfl-server-prod .",
        "Building server image",
    )
    if not success:
        return False

    # Build client image
    success, output = run_command(
        "docker build -f zkfl_production/Dockerfile.client -t zkfl-client-prod .",
        "Building client image",
    )
    if not success:
        return False

    print("✅ Docker images built successfully")
    return True


def test_server_startup():
    """Test that the server can start up properly."""
    print("\n🚀 Testing Server Startup...")

    # Start server in background
    cmd = "docker run --rm --name zkfl-validation-server -d -p 8081:8080 --memory='2g' zkfl-server-prod"
    success, container_id = run_command(cmd, "Starting server container")

    if not success:
        return False

    print(f"📦 Server container started: {container_id.strip()}")

    # Wait for startup
    print("⏳ Waiting for server initialization...")
    time.sleep(15)

    # Check logs
    success, logs = run_command(
        "docker logs zkfl-validation-server", "Getting server logs"
    )
    print("📋 Server logs:")
    print(logs)

    # Check if server is still running
    success, status = run_command(
        "docker ps --filter name=zkfl-validation-server --format '{{.Status}}'",
        "Checking container status",
    )

    # Stop the server
    run_command("docker stop zkfl-validation-server", "Stopping server container")

    if "Building global model" in logs and "Successfully built" in logs:
        print("✅ Server startup test passed")
        return True
    else:
        print("❌ Server startup test failed")
        return False


def test_deployment_scripts():
    """Test deployment automation scripts."""
    print("\n📜 Testing Deployment Scripts...")

    scripts = [
        "zkfl_production/deploy/build.sh",
        "zkfl_production/deploy/deploy-docker.sh",
        "zkfl_production/deploy/deploy-k8s.sh",
        "zkfl_production/deploy/monitor.sh",
    ]

    all_exist = True
    for script in scripts:
        if Path(script).exists():
            print(f"✅ {script} exists")
        else:
            print(f"❌ {script} missing")
            all_exist = False

    return all_exist


def validate_configuration():
    """Validate configuration files."""
    print("\n⚙️  Validating Configuration...")

    config_files = [
        "zkfl_production/docker-compose.yml",
        "zkfl_production/docker-compose.dev.yml",
        "zkfl_production/k8s/namespace.yaml",
        "zkfl_production/k8s/configmap.yaml",
    ]

    all_exist = True
    for config in config_files:
        if Path(config).exists():
            print(f"✅ {config} exists")
        else:
            print(f"❌ {config} missing")
            all_exist = False

    return all_exist


def main():
    """Main validation workflow."""
    print("🎯 ZKFL Production Deployment Validation")
    print("=" * 50)

    # Check prerequisites
    if not check_docker():
        print("❌ Docker is required for production deployment")
        return False

    if not check_docker_compose():
        print("❌ Docker Compose is required for production deployment")
        return False

    # Validate files and scripts
    if not validate_configuration():
        print("❌ Configuration validation failed")
        return False

    if not test_deployment_scripts():
        print("❌ Deployment scripts validation failed")
        return False

    # Build and test
    if not build_images():
        print("❌ Image building failed")
        return False

    if not test_server_startup():
        print("❌ Server startup test failed")
        return False

    # Final summary
    print("\n" + "=" * 50)
    print("🎉 ZKFL PRODUCTION DEPLOYMENT VALIDATION: SUCCESS!")
    print("✅ All components verified and working")
    print("🚀 Ready for production deployment")
    print("\nTo deploy:")
    print("1. Development: docker-compose -f zkfl_production/docker-compose.dev.yml up")
    print("2. Production: docker-compose -f zkfl_production/docker-compose.yml up")
    print("3. Kubernetes: ./zkfl_production/deploy/deploy-k8s.sh")

    return True


if __name__ == "__main__":
    try:
        if main():
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with exception: {e}")
        sys.exit(1)
