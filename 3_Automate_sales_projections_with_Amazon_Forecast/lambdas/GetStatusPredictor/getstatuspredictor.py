class SatusActive(ValueError):
    """Error thrown if previous job still in progress (catch & retry this in your SFn)"""
    pass

def lambda_handler(event, context):
    import boto3
    forecast = boto3.client('forecast')
    
    response = forecast.describe_predictor(
        PredictorArn=event['PredictorArn']
    )
    
    if response['Status'] != 'ACTIVE':
        raise SatusActive(f"previous job is running.")

    return event