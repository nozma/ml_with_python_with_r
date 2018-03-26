FROM nozma/ml-python-notebook-r

# Install R packages
RUN install2.r --error --deps TRUE GGally
RUN apt-get -y install libx11-dev
RUN apt-get -y install freeglut3 freeglut3-dev libgl1-mesa-dev libglu1-mesa-dev mesa-common-dev x11proto-gl-dev
RUN install2.r --error --deps TRUE mlr
RUN installGithub.r mlr-org/mlrCPO

CMD ["/init"]
