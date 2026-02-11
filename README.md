### Installation
#### Clone Repository
```
git clone git@github.com:laurencia7/migas-be-ai.git
```
#### Create Virtual Environment
```
python -m venv .venv
```
#### Activate Virtual Environment
```
# Windows
.venv\Scripts\activate

#Linux
source .venv/bin/activate
```
#### Install the packages
```
pip install -R requirements.txt
```

### Usage
#### Run App
```
uvicorn main:app --host 0.0.0.0 --port 8080
```