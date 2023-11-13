ARG BASE_IMAGE="nvcr.io/nvidia/driver:535-5.15.0-1040-nvidia-ubuntu22.04"
FROM $BASE_IMAGE


ARG APT_COMMAND="apt-get -o Acquire::Retries=3 -y"
ARG USER_ID
ARG USER_GROUP_ID
ENV DEBIAN_FRONTEND=noninteractive

RUN ${APT_COMMAND} update && ${APT_COMMAND} install -y wget \
    sudo \
    build-essential \
    checkinstall\
    zlib1g \
    zlib1g-dev \
    libssl-dev\
    libffi-dev\
    libbz2-dev \
    libsdl-image1.2-dev \
    libsdl-mixer1.2-dev \
    libsdl-ttf2.0-dev \
    libsdl1.2-dev \
    libsmpeg-dev \
    subversion \
    libportmidi-dev \
    ffmpeg \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev \
    libfreetype6-dev \
    libopenblas-dev

# Install dependencies
# add deadsnakes ppa then install python3.10

RUN ${APT_COMMAND} update && \
    ${APT_COMMAND} install software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    ${APT_COMMAND} update && \
    ${APT_COMMAND} install python3.10 python3.10-distutils python3.10-dev python3.10-venv && \
    ${APT_COMMAND} clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1


# remove old python3 symlink and recreate it pointing to python3.10
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.10 /usr/bin/python3

# Install python3 pip
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.10 get-pip.py

RUN ${APT_COMMAND} update && ${APT_COMMAND} install -y \
    xvfb \
    unzip \
    subversion \
    x11-apps \
    htop \
    curl \
    vim \
    git \
    tmux \
    mesa-utils \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    openssh-server \
    ssh \
    dbus \
    notification-daemon



ARG CHROME_VERSION="114.0.5735.90-1"
ARG CHROMEDRIVER_VERSION="114.0.5735.90"
RUN wget --no-verbose -O /tmp/chrome.deb https://dl.google.com/linux/chrome/deb/pool/main/g/google-chrome-stable/google-chrome-stable_${CHROME_VERSION}_amd64.deb \
  && apt install -y /tmp/chrome.deb \
  && rm /tmp/chrome.deb
RUN wget --no-verbose -O /tmp/chromedriver.zip https://chromedriver.storage.googleapis.com/${CHROMEDRIVER_VERSION}/chromedriver_linux64.zip \
  && unzip /tmp/chromedriver.zip -d /usr/bin \
  && rm /tmp/chromedriver.zip \
  && chmod 755 /usr/bin/chromedriver
ENV PATH=$PATH:/usr/bin/chromedriver

# Enable X11 forwarding for docker
RUN echo "X11UseLocalhost no" >> /etc/ssh/sshd_config && \
    echo "X11DisplayOffset 10" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config

# Add User for safely using mpi4py
RUN groupadd -g $USER_GROUP_ID user_group
RUN groupadd -g 998 docker

RUN useradd -u $USER_ID -g $USER_GROUP_ID -G 998 -ms /bin/bash user

# Granting sudo privileges
RUN usermod -aG sudo user
RUN echo "user:password" | chpasswd
RUN echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir /home/user/.ssh
RUN chmod 700 /home/user/.ssh
RUN touch /home/user/.ssh/authorized_keys
RUN chmod 600 /home/user/.ssh/authorized_keys
ADD .ssh/id_rsa.pub /home/user/.ssh/authorized_keys

# Add /home/user/.local/bin to PATH
ENV PATH="/home/user/.local/bin:${PATH}"

RUN echo "Port 2020" >> /etc/ssh/sshd_config
EXPOSE 2020


USER user

ENTRYPOINT sudo service ssh restart && bash