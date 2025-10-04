FROM rocker/r-base:4.3.2

# System libs needed by R packages
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev libssl-dev libxml2-dev libxt6 libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install CRAN packages
RUN R -e "install.packages(c('jsonlite','data.table','readr','dplyr','frbs','ggplot2'), repos='https://cloud.r-project.org')"

ENV R_LIBS_USER=/usr/local/lib/R/site-library
