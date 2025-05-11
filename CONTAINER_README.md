# Play Whe Prediction System - Containerized Environment

This guide explains how to run the Play Whe Prediction System using Podman containers.

## Prerequisites

- Podman installed (already installed on Fedora by default)
- podman-compose (optional for docker-compose.yml support)

## Quick Start

We've provided a helper script `podman-helper.sh` to make working with containers easier.

### Build the container image

```bash
./podman-helper.sh build
```

### Generate sample data

```bash
./podman-helper.sh shell
# Inside the container shell:
python data/create_sample_data.py
exit
```

### Run data processing

```bash
./podman-helper.sh process
```

### Build prediction models

```bash
./podman-helper.sh models
```

### Run a prediction

```bash
./podman-helper.sh predict --period morning --top 5
```

### Run tests

```bash
./podman-helper.sh test
```

## Advanced Usage

### Using podman-compose

If you have podman-compose installed, you can use the docker-compose.yml file:

```bash
# Install podman-compose if needed
sudo dnf install podman-compose

# Start the services
podman-compose up -d

# Run a command in the app container
podman exec -it playwhe-app python make_prediction.py --period morning --top 5

# Stop the services
podman-compose down
```

### Manual Container Usage

If you prefer to work directly with Podman commands:

```bash
# Build the image
podman build -t playwhe-prediction -f Containerfile .

# Run the container with interactive shell
podman run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/predictions:/app/predictions" \
  -v "$(pwd)/analysis:/app/analysis" \
  playwhe-prediction bash

# Run a specific command
podman run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/predictions:/app/predictions" \
  -v "$(pwd)/analysis:/app/analysis" \
  playwhe-prediction python make_prediction.py --period morning --top 5
```

## Container Structure

- The container uses Python 3.10 as a base image
- All project files are copied to /app inside the container
- Data directories are mounted as volumes for persistence
- The container runs as a non-root user for security

## Common Issues and Troubleshooting

### Permission Denied on Mounted Volumes

If you encounter permission issues with mounted volumes, you may need to adjust SELinux contexts:

```bash
# Add :Z to volume mounts
podman run --rm -it \
  -v "$(pwd)/data:/app/data:Z" \
  -v "$(pwd)/logs:/app/logs:Z" \
  playwhe-prediction bash
```

### Network Issues

For scraping operations that require network access:

```bash
# Run with host network
podman run --rm --network=host \
  -v "$(pwd)/data:/app/data" \
  playwhe-prediction python nlcb_scraper.py
```

## Best Practices

1. Use volumes for persistent data
2. Run containers with the least privileges needed
3. Use separate containers for different concerns
4. Keep containers stateless where possible
5. Use environment variables for configuration

## Cleanup

To remove all containers and images related to this project:

```bash
# Remove containers
podman ps -a --filter "name=playwhe" -q | xargs podman rm -f

# Remove images
podman images --filter "reference=playwhe-prediction" -q | xargs podman rmi -f
```