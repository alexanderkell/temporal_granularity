FROM python:3.6


COPY requirements.txt /
RUN pip install -r requirements.txt

ADD . /app
COPY . /app

WORKDIR /app

# ENV NAME World

ENV PYTHONPATH "${PYTHONPATH}:/src"

ENTRYPOINT ["python", "-m", "scoop", "src/models/optimisation_algorithms/genetic_algorithms/nsga2.py"]
CMD [""]
# CMD ["python", "run/timing/batch_run_timer.py"]


