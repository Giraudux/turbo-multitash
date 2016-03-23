FROM debian

MAINTAINER Alexis Giraudet <https://github.com/Giraudux>

RUN apt-get -y update && \
    apt-get -y install build-essential git mpich openssh-server rsyslog
RUN useradd -m mpi && \
    echo mpi:mpi | chpasswd
RUN touch /var/log/user.log
RUN mkdir /var/run/sshd

EXPOSE 22

CMD ["/usr/sbin/rsyslogd"]
CMD ["/usr/sbin/sshd", "-D"]
CMD ["tail", "-f", "/var/log/user.log"]
