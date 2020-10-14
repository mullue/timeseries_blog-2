# Air Quality Forecasting Lab

The purpose of the labs is to demo how to use Amazon SageMaker built-in algorithm DeepAR to do time series data forecasting. Besides a notebook to walk through the steps, we provide ml pipeline creation reference.

## Lab Structure
There are two Jupyter Notebooks;  
1. [Forecasting Air Quality with Amazon SageMaker and DeepAR](./01_train_and_evaluate_air_quality_deepar_model.ipynb) to demo time series data forecasting.
2. [Air Quality Forecasting ML Pipeline (manual)](./02_manual_ml_pipeline_creation_for_air_quality_forecasting.ipynb) to demo ML Pipeline manual creation.

Especially, the second notebook is part of the demo of ML Pipeline creation for the first. To create the lab, please create a AWS CloudFormation stack with [CFN template](./air_quality_forecasting_ml_pipeline.yml).
 
## CloudFormation Stack Creation

> Assumed User is create the labs environment in AWS Sydney Region. If not, please correct the console link's query parameter `region`.

### Create Steps 

1. Refer to Workshop 'Launch Stack' link or download github repo - [timeseries_blog](https://github.com/glyfnet/timeseries_blog.git) and launch AWS CloudFormation stack with referring to local [CFN template](./air_quality_forecasting_ml_pipeline.yml). Please accept the default parameter values which will refer to CodeCommit repo to integrate with CodePipeline for ML Pipeline creation and execution.

![CloudFormation Stack Parameter Screen](./img/lab_desc_cfn_stack_parameters.png)

2. Once filling in stack name, click 'Next' button twice, then, scroll down to bottom of the page. Check ***I acknowledge that AWS CloudFormation might create IAM resources with custom names.*** and Click 'Click stack' button. The whole stack creation takes approx. 5-8mins.

![CloudFormation Stack Creation Screen](./img/lab_desc_cfn_stack_creation.png)

3. Once CFN stack is created, you can explore pipeline setup in [AWS CodePipeline](https://console.aws.amazon.com/codesuite/codepipeline/pipelines?region=ap-southeast-2). Also, please note that CodePipeline will be automatically triggered once the stack  creation is done (as related CodeCommit project). 

![CodePipeline Console Screen](./img/lab_desc_codepipeline.png)

4. Meanwhile, you can explore the notebooks with [Amazon SageMaker Notebook instances](https://console.aws.amazon.com/sagemaker/home?region=ap-southeast-2#/notebook-instances). Then, click 'Open Jupyter' or 'Open JupyterLab' to view the notebooks.

![Amazon SageMaker Notebook Instances Screen](./img/lab_desc_sagemaker_notebook_instances.png)

1. For example, once 'Open Jupyter', you can start playing with the notebook. Highly recommend that you should start with [Forecasting Air Quality with Amazon SageMaker and DeepAR](./01_train_and_evaluate_air_quality_deepar_model.ipynb).

![Jupyter Notebooks Screen](./img/lab_desc_lab_jupyter_notebooks.png)

## Pipeline Design

![Pipline Design](./img/aqf-ml-pipeline-design.png)

### Description

* With default CFN parameter values setup during CFN stack creation, [GitHub Repo - timeseries_blog](https://github.com/glyfnet/timeseries_blog) will be mirrored to CodeCommit repo so that user can experiment code change to trigger pipeline easily.
* CodePipeline pipeline orchestrates the build process with CodeBuild project.
* CodeBuild project process `preprocess` container build and ML Pipeline creation & execution with Step Functions state machine. (***the workflow won't be part of CFN stack, hence, you may manually remove it while deleting CFN stack.***)
* State machine demo ML pipeline and orchestrate data preprocessing, model training/tuning and batch transform.
* Amazon SageMaker Notebook instance can be used to explore notebooks.