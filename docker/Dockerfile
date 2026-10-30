# syntax=docker/dockerfile:1
FROM ros:humble-ros-base AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip python3-colcon-common-extensions ros-humble-rosbag2 \
      ros-humble-rosbag2-storage-mcap ros-humble-tf2-ros ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY docker/requirements-base.txt /tmp/requirements-base.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements-base.txt

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV PYTHONPATH=/workspace/src
WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

FROM base AS sim
RUN apt-get update && apt-get install -y --no-install-recommends \
      git libosmesa6-dev libgl1-mesa-dev libglu1-mesa libglew-dev libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*
COPY docker/requirements-sim.txt /tmp/requirements-sim.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements-sim.txt \
 && pip3 install --no-cache-dir --no-deps robosuite==1.4.0 \
 && pip3 install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
ARG LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO \
 && cd /opt/LIBERO && git checkout ${LIBERO_COMMIT} \
 && pip3 install --no-cache-dir --no-deps -e .
ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_CONFIG_PATH=/opt/libero_config
RUN echo n | python3 -c "import libero.libero"
