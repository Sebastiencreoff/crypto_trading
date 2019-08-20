FROM centos:7

RUN yum update -y
RUN yum install git -y

WORKDIR /home/app
RUN git clone https://github.com/Sebastiencreoff/crypto_trading.git /home/app/

