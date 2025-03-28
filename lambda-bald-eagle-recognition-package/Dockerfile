# Use an official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.9

# Copy function code and requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
COPY app.py ${LAMBDA_TASK_ROOT}/
COPY cookies.txt ${LAMBDA_TASK_ROOT}/


# Install dependencies from requirements.txt
# Using --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Set the CMD to your handler (app.lambda_handler)
# The format is <script_name>.<handler_function>
CMD ["app.lambda_handler"]