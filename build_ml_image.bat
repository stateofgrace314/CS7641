docker build -t kgrace/ml_anaconda -f Docker/Dockerfile.ml_anaconda .
docker build -t kgrace/ml_latex -f Docker/Dockerfile.ml_latex .
docker build -t kgrace/ml_cudnn -f Docker/Dockerfile.ml_cudnn .
docker build -t kgrace/ml_dev -f Docker/Dockerfile.ml_dev .