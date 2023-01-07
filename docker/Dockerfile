ARG WEB_NAV_DIR="../"
ARG BASE_IMAGE="nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04"
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y wget \
    build-essential \
    checkinstall \
    zlib1g \
    xvfb \
    sudo \
    unzip \
    libssl-dev \
    subversion \
    x11-apps \
    htop \
    curl \
    vim \
    git \
    tmux \
    libffi-dev \
    mesa-utils \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    openssh-server \
    ssh \
    dbus \
    zlib1g-dev \
    notification-daemon

RUN wget https://www.python.org/ftp/python/3.10.8/Python-3.10.8.tgz
RUN tar -xvf Python-3.10.8.tgz
WORKDIR Python-3.10.8
RUN ./configure
RUN make
RUN make install
WORKDIR /

# Symlink pip3 to pip
RUN ln -s /usr/local/bin/pip3 /usr/local/bin/pip
RUN pip3 install --upgrade pip

# Install google chrome and chromedriver
ARG CHROME_VERSION="109.0.5414.74-1"
ARG CHROMEDRIVER_VERSION="109.0.5414.74"
RUN wget --no-verbose -O /tmp/chrome.deb https://dl.google.com/linux/chrome/deb/pool/main/g/google-chrome-stable/google-chrome-stable_${CHROME_VERSION}_amd64.deb \
  && apt install -y /tmp/chrome.deb \
  && rm /tmp/chrome.deb
RUN wget --no-verbose -O /tmp/chromedriver.zip https://chromedriver.storage.googleapis.com/${CHROMEDRIVER_VERSION}/chromedriver_linux64.zip \
  && unzip /tmp/chromedriver.zip -d /usr/bin \
  && rm /tmp/chromedriver.zip \
  && chmod 755 /usr/bin/chromedriver
ENV PATH=$PATH:/usr/bin/chromedriver

RUN svn export https://github.com/google-research/google-research/trunk/compositional_rl && mv /compositional_rl /web_nav

# Install bootstrap files
RUN mkdir -p /web_nav/gwob/bootstrap && cd /web_nav/gwob/bootstrap && \
  wget https://github.com/twbs/bootstrap/releases/download/v4.3.1/bootstrap-4.3.1-dist.zip && \
  unzip bootstrap-4.3.1-dist.zip  && \
  cp bootstrap-4.3.1-dist/css/bootstrap.min.css . && \
  cp bootstrap-4.3.1-dist/js/bootstrap.min.js . && \
  rm -r bootstrap-4.3.1-dist*

# Install MiniWob
WORKDIR /web_nav
RUN git clone https://github.com/stanfordnlp/miniwob-plusplus gwob/miniwob-plusplus
RUN cd gwob/miniwob-plusplus && git checkout 833a477a8fbfbd2497e95fee019f76df2b9bd75e
RUN pip install 2to3
RUN 2to3 gwob/miniwob-plusplus/python/miniwob -w

RUN python3 integrate_miniwob.py
RUN pip install gwob/miniwob-plusplus/python/

# Install gMiniWob
RUN pip install gwob/
RUN pip install --force-reinstall numpy==1.23.5

# Create user (need a non-root user for running google-chrome without --no-sandbox)
RUN useradd -ms /bin/bash user
RUN usermod -aG sudo user
RUN passwd -d user

# Enable X11 forwarding for docker
RUN echo "X11UseLocalhost no" >> /etc/ssh/sshd_config
RUN echo "X11DisplayOffset 10" >> /etc/ssh/sshd_config

USER user

RUN mkdir /home/user/.ssh
RUN chmod 700 /home/user/.ssh
RUN touch /home/user/.ssh/authorized_keys
RUN chmod 600 /home/user/.ssh/authorized_keys
ADD $WEB_NAV_DIR/.ssh/id_rsa.pub /home/user/.ssh/authorized_keys

EXPOSE 22

ENTRYPOINT export DBUS_SESSION_BUS_ADDRESS=`dbus-daemon --fork --config-file=/usr/share/dbus-1/session.conf --print-address` && \
    sudo service dbus start && \
    sudo service ssh start && \
    exec bash