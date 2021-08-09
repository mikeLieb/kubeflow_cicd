FROM google/cloud-sdk:latest

RUN mkdir modeling && \
    cd modeling && \
    git clone https://github.com/mikeLieb/kubeflow_cicd.git

COPY train.sh ./

ENTRYPOINT ["bash", "./train.sh"]
