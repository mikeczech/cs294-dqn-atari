provider "aws" {
  access_key = "${var.access_key}"
  secret_key = "${var.secret_key}"
  region     = "us-east-1"
}

resource "aws_s3_bucket" "bucket" {
  bucket = "rl-artifacts-bucket"
  acl    = "private"

  tags {
    Name = "RL Artifacts"
    Environment = "Dev"
  }
}

resource "aws_iam_role" "s3_iam_role" {
  name = "s3_iam_role"
  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Effect": "Allow"
    }
  ]
}
EOF
}

resource "aws_iam_instance_profile" "s3_instance_profile" {
  name = "s3_instance_profile"
  role = "${aws_iam_role.s3_iam_role.name}"
}

resource "aws_iam_role_policy" "s3_iam_role_policy" {
  name = "s3_iam_role_policy"
  role = "${aws_iam_role.s3_iam_role.id}"

  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3::::rl-artifacts-bucket",
        "arn:aws:s3::::rl-artifacts-bucket/*"
      ]
    }
  ]
}
EOF
}

resource "aws_vpc" "default" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_internet_gateway" "default" {
  vpc_id = "${aws_vpc.default.id}"
}

resource "aws_route" "internet_access" {
  route_table_id         = "${aws_vpc.default.main_route_table_id}"
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = "${aws_internet_gateway.default.id}"
}

resource "aws_subnet" "default" {
  vpc_id                  = "${aws_vpc.default.id}"
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
}

resource "aws_security_group" "default" {
  name   = "default_security_group"
  vpc_id = "${aws_vpc.default.id}"

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Tensorboard access
  ingress {
    from_port   = 6006
    to_port     = 6006
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_key_pair" "auth" {
  public_key = "${file(var.ssh_public_key)}"
}

resource "aws_instance" "deep_learning" {
  ami           = "ami-ca0136b0"
  instance_type = "p2.xlarge"
  key_name      = "${aws_key_pair.auth.id}"

  vpc_security_group_ids = ["${aws_security_group.default.id}"]

  subnet_id = "${aws_subnet.default.id}"

  iam_instance_profile = "${aws_iam_instance_profile.s3_instance_profile.id}"
}

output "ip" {
  value = "${aws_instance.deep_learning.public_ip}"
}
