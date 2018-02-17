import sys
from dqn import dqn_atari as atari


def main():
    if len(sys.argv) > 1:
        atari.run(sys.argv[1])
    else:
        print("Please provide a checkpoint")

if __name__ == "__main__":
    main()
