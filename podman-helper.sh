#!/bin/bash
# Helper script for managing the Play Whe Prediction System with Podman

# Variables
IMAGE_NAME="playwhe-prediction"
CONTAINER_NAME="playwhe-app"
VOLUME_NAME="playwhe-data"

# Function to display usage information
show_usage() {
  echo "Usage: $0 [command]"
  echo ""
  echo "Commands:"
  echo "  build       - Build the container image"
  echo "  run         - Run the container in interactive mode"
  echo "  predict     - Run a prediction with specified parameters"
  echo "  scrape      - Run the scraper to collect data"
  echo "  process     - Process the collected data"
  echo "  models      - Build the prediction models"
  echo "  test        - Run tests on the system"
  echo "  shell       - Open a shell in the container"
  echo "  stop        - Stop the running container"
  echo "  help        - Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0 build"
  echo "  $0 predict --period morning --top 10"
  echo "  $0 shell"
}

# Function to build the container image
build_image() {
  echo "Building container image: $IMAGE_NAME"
  podman build -t $IMAGE_NAME -f Containerfile .
}

# Function to run the container interactively
run_container() {
  echo "Running container: $CONTAINER_NAME"
  podman run --name $CONTAINER_NAME -it --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/logs:/app/logs" \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/predictions:/app/predictions" \
    -v "$(pwd)/analysis:/app/analysis" \
    $IMAGE_NAME bash
}

# Function to run a prediction
run_prediction() {
  echo "Running prediction with args: $@"
  podman run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/logs:/app/logs" \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/predictions:/app/predictions" \
    -v "$(pwd)/analysis:/app/analysis" \
    $IMAGE_NAME python make_prediction.py "$@"
}

# Function to run the scraper
run_scraper() {
  echo "Running scraper with args: $@"
  podman run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/logs:/app/logs" \
    $IMAGE_NAME python nlcb_scraper.py "$@"
}

# Function to process data
process_data() {
  echo "Processing data with args: $@"
  podman run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/logs:/app/logs" \
    $IMAGE_NAME python merge_data.py "$@"
}

# Function to build models
build_models() {
  echo "Building models with args: $@"
  podman run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/logs:/app/logs" \
    -v "$(pwd)/models:/app/models" \
    $IMAGE_NAME python prediction_models.py "$@"
}

# Function to run tests
run_tests() {
  echo "Running tests with args: $@"
  podman run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/logs:/app/logs" \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/predictions:/app/predictions" \
    -v "$(pwd)/analysis:/app/analysis" \
    $IMAGE_NAME python test_enhanced_models.py "$@"
}

# Function to open a shell in the container
run_shell() {
  echo "Opening shell in container"
  podman run --rm -it \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/logs:/app/logs" \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/predictions:/app/predictions" \
    -v "$(pwd)/analysis:/app/analysis" \
    $IMAGE_NAME bash
}

# Function to stop the container
stop_container() {
  echo "Stopping container: $CONTAINER_NAME"
  podman stop $CONTAINER_NAME 2>/dev/null || echo "No container running"
}

# Process command line arguments
case "$1" in
  build)
    build_image
    ;;
  run)
    run_container
    ;;
  predict)
    shift
    run_prediction "$@"
    ;;
  scrape)
    shift
    run_scraper "$@"
    ;;
  process)
    shift
    process_data "$@"
    ;;
  models)
    shift
    build_models "$@"
    ;;
  test)
    shift
    run_tests "$@"
    ;;
  shell)
    run_shell
    ;;
  stop)
    stop_container
    ;;
  help|*)
    show_usage
    ;;
esac