FROM python:3.11-slim-buster

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/opt/app:${PYTHONPATH}"

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libgl1-mesa-glx \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

ENV MPLCONFIGDIR=/opt/app/data/matplotlib_cache
RUN mkdir -p ${MPLCONFIGDIR} && chmod 777 ${MPLCONFIGDIR}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3000

COPY ./app ./app   
COPY ./configs ./configs 

CMD ["python", "-m", "app.cli", "--help"]