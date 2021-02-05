To build customized container image with training code and additional libraries: 
	
	bash build_dlc_sagemaker.sh <image_name> <tag_name> <aws_account_id>	

Push the customized docker image to ECR:
	
	bash push_dlc_sagemaker.sh <image_name> <tag_name> <aws_account_id>	
	
Example:
	
	bash build_dlc_sagemaker.sh sagemaker test 12345
	bash push_dlc_sagemaker.sh sagemaker test 12345

After push the image to your own ECR, example command to run:
	
	pip install sagemaker==2.23.3
	python run_train.py --bucket_name <your_s3_bucket_name_store_data> \
		    	    --role 'arn:aws:iam::<aws_account_id>:role/service-role/AmazonSageMaker-ExecutionRole-2xxxxxx' \
		   	    --image_uri '<aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/sagemaker:test_0.1'

