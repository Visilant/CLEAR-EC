#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DOCKER_IMAGE_TAG="clear_ec_algorithm"

echo ""
echo "= STEP 1 = (Re)build the image"
export DOCKER_QUIET_BUILD=1
source "${SCRIPT_DIR}/do_build.sh"
echo "==== Done"
echo ""

build_timestamp=$( docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")

if [ -z "$build_timestamp" ]; then
    echo "Error: Failed to retrieve build information for container $DOCKER_IMAGE_TAG"
    exit 1
fi

formatted_build_info=$(echo $build_timestamp | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')

output_filename="${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"
output_path="${SCRIPT_DIR}/$output_filename"

echo "= STEP 2 = Saving the image"
echo "This can take a while."

# Grand Challenge requires the legacy Docker tar format (manifest.json at the
# root). On Docker 25+ `docker save` emits OCI archives (blobs/sha256/...) which
# GC rejects, and the default `docker` buildx driver cannot export the docker
# format either. Use a `docker-container` buildx builder which supports it
# regardless of the daemon's image store.
BUILDER_NAME="clearec-builder"
if ! docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
    echo "Creating buildx docker-container builder '$BUILDER_NAME'..."
    docker buildx create --name "$BUILDER_NAME" --driver=docker-container >/dev/null
fi

echo "Re-exporting via buildx in legacy Docker format..."
intermediate_tar="${SCRIPT_DIR}/${DOCKER_IMAGE_TAG}.tar"
docker buildx build \
    --builder "$BUILDER_NAME" \
    --platform=linux/amd64 \
    --tag "$DOCKER_IMAGE_TAG" \
    --output "type=docker,dest=${intermediate_tar}" \
    "$SCRIPT_DIR"

if ! tar -tf "$intermediate_tar" | grep -q '^manifest.json$'; then
    echo "ERROR: buildx export did not produce a legacy Docker tar."
    echo "Contents:"
    tar -tf "$intermediate_tar" | head -5
    exit 1
fi

gzip -c "$intermediate_tar" > "$output_path"
rm -f "$intermediate_tar"

printf "Saved as: \e[32m${output_filename}\e[0m\n"

echo "==== Done"
echo ""


echo "= STEP 3 = Packing the model"
echo "This can take a while."
output_tarball_name="${SCRIPT_DIR}/model.tar.gz"

tar -czf $output_tarball_name -C "${SCRIPT_DIR}/model" .
printf "Saved as: \e[32mmodel.tar.gz\e[0m\n"

echo "==== Done"
echo ""


printf "\e[31mIMPORTANT: Please upload the model.tar.gz as separate Model to your Algorithm!\e[0m\n"
