# Base Image
ARG BASE_IMAGE="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04"
FROM $BASE_IMAGE

# Environment Variables
ARG APT_COMMAND="apt-get -o Acquire::Retries=3 --no-install-recommends -y"
ARG USER_ID=1000
ARG USER_GROUP_ID=1000
ARG CHROME_VERSION="114.0.5735.90-1"
ARG CHROMEDRIVER_VERSION="114.0.5735.90"
ENV DEBIAN_FRONTEND=noninteractive

# Add the Deadsnakes PPA for Python 3.10 and Package Installation
RUN ${APT_COMMAND} update && \
    ${APT_COMMAND} install software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    ${APT_COMMAND} update && \
    ${APT_COMMAND} upgrade && \
    ${APT_COMMAND} install \
    python3.10 \
    python3.10-dev  \
    python3.10-venv  \
    python3.10-distutils \
    wget \
    sudo \
    build-essential  \
    ssh \
    openssh-server \
    libnss3  \
    dbus  \
    dbus-x11 \
    unzip  \
    x11-apps  \
    x11-utils && rm -rf /var/lib/apt/lists/*

# Add TensorRT
RUN ${APT_COMMAND} update && \
    ${APT_COMMAND} install python3-libnvinfer-dev \
    python3-libnvinfer

# Install pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Update Python Alternatives to use Python 3.10 as the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --set python /usr/bin/python3.10

# Chrome Installation
RUN wget --no-verbose -O /tmp/chrome.deb https://dl.google.com/linux/chrome/deb/pool/main/g/google-chrome-stable/google-chrome-stable_${CHROME_VERSION}_amd64.deb && \
    ${APT_COMMAND} update && \
    ${APT_COMMAND} --fix-broken install && \
    ${APT_COMMAND} install /tmp/chrome.deb && \
    rm /tmp/chrome.deb

# SSH and X11 Configuration
RUN echo "X11UseLocalhost no\nX11DisplayOffset 10\nPasswordAuthentication yes\nPort 2020" >> /etc/ssh/sshd_config && \
    mkdir /run/sshd

# User Creation
RUN groupadd -g $USER_GROUP_ID user_group && \
    groupadd -g 998 docker && \
    useradd -u $USER_ID -g $USER_GROUP_ID -G 998 -ms /bin/bash user && \
    usermod -aG sudo user && \
    echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    mkdir -p /home/user/.ssh && \
    chmod 700 /home/user/.ssh && \
    touch /home/user/.ssh/authorized_keys && \
    chmod 600 /home/user/.ssh/authorized_keys

# Switch to User
USER user

# Set Work Directory
WORKDIR /home/user

# Create and Activate a Virtual Environment
RUN python3.10 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip setuptools

COPY .ssh/id_rsa.pub /home/user/.ssh/authorized_keys
RUN sudo chown -R user:user_group /home/user

# Set PATH to include the virtual environment's bin directory
ENV PATH="/home/user/venv/bin:${PATH}"
#ENV PATH="/usr/local/cuda/bin:${PATH}"
#ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Entry Point
ENTRYPOINT sudo service ssh start && sudo service dbus start && bash
