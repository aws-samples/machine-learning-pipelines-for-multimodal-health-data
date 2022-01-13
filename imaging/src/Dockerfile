FROM public.ecr.aws/docker/library/python:3.7-slim-buster

COPY ./requirements.txt /opt/
RUN pip3 install --no-cache-dir -r /opt/requirements.txt
ENV PYTHONUNBUFFERED=TRUE
ENV AWS_DEFAULT_REGION=##REGION##
ENV PYTHONPATH "/opt/:${PYTHONPATH}"

COPY ./dcm2nifti_processing.py /opt/
COPY ./radiomics_utils.py /opt/

ENTRYPOINT ["python3", "/opt/dcm2nifti_processing.py"]
