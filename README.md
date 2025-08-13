# E2E News Recommendation System

## Work flow
- config.yaml
- config/configuration.py
- components
- pipeline
- main.py
- app.py


## How to run?
### STEPS:
Clone the repository
```bash
git clone https://github.com/kanewyp/end2end-news-recommendation-system
```
### STEP 01- Create a conda environment after opening the repository
```bash
conda create -n project python=3.8 -y
```

```bash
conda activate project
```

### STEP 02- Install the requirements
```bash
pip install -r requirements.txt
```

### STEP 03- Run on streamlit
```bash
streamlit run app.py
```

## Streamlit app Docker Image Deployment
### 1. Login with you AWS console and launch an EC2 instance
### 2. Run the following commands

```bash
sudo apt-get update -y

sudo apt-get upgrade

#Install Docker

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
```

```bash
git clone "your-project"
```

```bash
docker build -t entbappy/stapp:latest . 
```

```bash
docker images -a  
```

```bash
docker run -d -p 8501:8501 entbappy/stapp 
```

```bash
docker ps  
```

```bash
docker stop container_id
```

```bash
docker rm $(docker ps -a -q)
```

```bash
docker login 
```

```bash
docker push entbappy/stapp:latest 
```

```bash
docker rmi entbappy/stapp:latest
```

```bash
docker pull entbappy/stapp
```