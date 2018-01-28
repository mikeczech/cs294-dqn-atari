import socket
import dqn_atari as atari


def main():
    print("Running Experiments on {}".format(socket.gethostname()))
    atari.run()

if __name__ == "__main__":
    main()
