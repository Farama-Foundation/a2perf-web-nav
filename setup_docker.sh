# if WEB_NAV_DIR variable is not set, then set it to the current working directory
if [ -z "$WEB_NAV_DIR" ]; then
  WEB_NAV_DIR=$(pwd)
fi

if [ -z "$WEB_NAV_CONTAINER_NAME" ]; then
  WEB_NAV_CONTAINER_NAME=web_nav_container
fi

echo "WEB_NAV_DIR: $WEB_NAV_DIR"
echo "WEB_NAV_CONTAINER_NAME: $WEB_NAV_CONTAINER_NAME"

# create ssh-key in WEB_NAV_DIR without password
mkdir -p "$WEB_NAV_DIR/.ssh"
ssh-keygen -t rsa -b 4096 -C "web_nav" -f "$WEB_NAV_DIR/.ssh/id_rsa" -N ""

# CONNECT TO RUNNING DOCKER CONTAINER WITH THIS COMMAND
# ssh user@127.0.0.1 -p 2022 -i .ssh/id_rsa

# Set the environment variable for where the project is located
docker build --build-arg WEB_NAV_DIR="$WEB_NAV_DIR" -f $WEB_NAV_DIR/docker/Dockerfile -t rlperf/web_nav:latest .

# If you do not have administrator rights to the host machine, remove the privileged command. Also do this if you only want to run headless mode.
xhost + local: # Allow docker to access the display
docker run --rm \
  -it \
  -p 2022:22 \
  -e DISPLAY="$DISPLAY" \
  -v "$HOME/.Xauthority:/user/.Xauthority:rw" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --name $WEB_NAV_CONTAINER_NAME \
  --workdir=/web_nav \
  --privileged \
  --gpus=all \
  rlperf/web_nav:latest
