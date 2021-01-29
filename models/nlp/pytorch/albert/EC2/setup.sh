mkdir -p /home/ubuntu/data/wiki_train
mkdir -p /home/ubuntu/data/wiki_validation
mkdir -p /home/ubuntu/data/wiki_demo
mkdir -p /home/ubuntu/data/squad
mkdir -p /home/ubuntu/output

aws configure set aws_access_key_id $1
aws configure set aws_secret_access_key $2
aws configure set default.region us-east-1
aws configure set region us-east-1
pip install torch==1.6.0
pip install tensorboard
#data install
aws s3 sync s3://yuliu-dev-east/wiki_demo "/home/ubuntu/data/wiki_train"
aws s3 sync s3://yuliu-dev-east/wiki_demo "/home/ubuntu/data/wiki_validation"
aws s3 sync s3://yuliu-dev-east/wiki_demo "/home/ubuntu/data/wiki_demo"
aws s3 sync s3://yuliu-dev-east/squad "/home/ubuntu/data/squad"
# install transformers
pip install transformers
