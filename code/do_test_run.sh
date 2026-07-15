#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="clear_ec_algorithm"

DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

cleanup() {
    echo "=+= Cleaning permissions ..."
    docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "chmod -R -f o+rwX /output/* || true"

    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null
}

# Allow the Docker user to read inputs and the model bundle.
chmod -R -f o+rX "$INPUT_DIR" "${SCRIPT_DIR}/model"

if [ -d "${OUTPUT_DIR}/interf0" ]; then
  chmod -f o+rwX "${OUTPUT_DIR}/interf0"

  echo "=+= Cleaning up any earlier output"
  docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "${OUTPUT_DIR}/interf0":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interf0"
fi


docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null

trap cleanup EXIT

# Use GPUs if the host's Docker daemon can offer them. On Grand Challenge the
# platform always provides GPUs; locally we just want a working smoke test.
GPU_ARGS=""
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia \
   || [ -e /usr/bin/nvidia-container-runtime ]; then
    GPU_ARGS="--gpus all"
    echo "=+= Detected NVIDIA container runtime; running with --gpus all"
else
    echo "=+= No NVIDIA container runtime detected; running on CPU (smoke test)"
fi

run_docker_forward_pass() {
    local interface_dir="$1"

    echo "=+= Doing a forward pass on ${interface_dir}"

    docker run --rm $GPU_ARGS \
        --platform=linux/amd64 \
        --network none \
        --volume "${INPUT_DIR}/${interface_dir}":/input:ro \
        --volume "${OUTPUT_DIR}/${interface_dir}":/output \
        --volume "$DOCKER_NOOP_VOLUME":/tmp \
        --volume "${SCRIPT_DIR}/model":/opt/ml/model:ro \
        "$DOCKER_IMAGE_TAG"

  echo "=+= Wrote results to ${OUTPUT_DIR}/${interface_dir}"
}


run_docker_forward_pass "interf0"


echo "=+= Save this image for uploading via ./do_save.sh"
