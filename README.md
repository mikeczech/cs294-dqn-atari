# Playing Atari with Deep Reinforcement Learning

This repository contains code for training a neural network to play Atari games.
All code is based on ideas of the DeepMind paper  [*Playing Atari with Deep Reinforcement Learning*](https://arxiv.org/pdf/1312.5602v1.pdf) and was created in the context of the course [*CS 294: Deep Reinforcement Learning, Fall 2017*](http://rll.berkeley.edu/deeprlcourse/) of the [Berkely University](https://www.berkeley.edu/).

In addition to training a network locally, the repository makes it easy to train on a GPU EC2 instance in AWS.

![Breakout](images/breakout.gif)

## Requirements

* Bash
* Python 3
* [virtualenv](https://virtualenv.pypa.io/en/stable/installation/)
* [terraform (for AWS deployment only)](https://www.terraform.io/)

## General Instructions

All operations can be accessed through the *go script* inside the root directory. Running the script without parameters prints available operations.

```bash
./go
Usage: ./go clean | deploy | ip | tensorboard | ssh | run | sync | gpu-usage | tf
```

## Training the Network

To start training, run

```bash
./go run src/train_dqn.py
```
The default game is *Breakout* and can be changed by modifying the file [src/dqn/dqn\_atari.py](src/dqn/dqn\_atari.py).
All parameters of the neural network are periodically saved during training and can be found in the folder *checkpoints/*.

To observe how the network develops during training, run
```bash
./go tensorboard
```
Then visit the displayed URL which points to a local [Tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) server which provides a nice graph showing the relationship between the number of training steps and the strength of the neural network in playing the game.

## Loading a Pretrained Network

To load a pretrained model, run
```bash
./go run src/run_dqn.py ${path to your checkpoint}
```
where the last parameter points to a checkpoint that was created during a previous training session.

## Training in AWS

To provision a ready-to-use GPU instance in AWS, first set the environment variables **SSH\_KEY**, **AWS\_ACCESS\_KEY**, and **AWS\_SECRET\_KEY**
```bash
export SSH_KEY=~/.ssh/id_rsa
export AWS_ACCESS_KEY={your access key}
export AWS_SECRET_KEY={your secret key}
```
Then run
```bash
./go deploy
./go sync
```
The first command will use Terraform and [Ansible](https://www.ansible.com/) to create a new EC2 instance based on the [Amazon Deep Learning Base AMI](https://docs.aws.amazon.com/dlami/latest/devguide/overview-base.html). Note that you might need to request a limit increase for p2.xlarge instances first. The second command synchronizes the remote machine with the local one.

To login to the freshly created EC2 instance, run
```bash
./go ssh
```

Now you are logged in to a [tmux](https://wiki.ubuntuusers.de/tmux/) environment which provides all the operations of this repository (e.g. run *./go run src/train\_dqn.py* to start training).

## Terminate all AWS Resources after Training

Simply run
```bash
./go tf destroy
```
and follow the dialog.
