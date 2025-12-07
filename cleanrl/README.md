# Curriculum Learning for Chess AI
PPO를 활용하여 체스 AI를 커리큘럼 학습 방식으로 훈련하는 프로젝트입니다.

[cleanrl](https://github.com/vwxyzjn/cleanrl)을 기반으로 하여 구현하였고, 환경은 [PettingZoo](https://pettingzoo.farama.org/environments/classic/chess/)의 체스 환경을 사용합니다.

## Setup
- ubuntu 22.04 or higher LST
- Python 3.10 (conda recommended)
  ```bash
  conda create -n chess_ai python=3.10
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage
아래의 명령어를 실행하기 전에 현재 directory가 code인지 확인하세요
### generate-data
```bash
python ppo_chess_clean.py --mode generate-data
```
- --num-games [Number]
  - 생성할 기보의 수 (default: 1000)
- --num-workers [Number]
  - 병렬로 실행할 작업자 수 (default: 1000)

### train
```bash
python ppo_chess_clean.py --mode curriculum
```
- --data-path [Path]
  - 생성된 체스 기보 파일 Path
- --use-wandb
  - wandb를 통해서 logging

### eval
```bash
python ppo_chess_clean.py --mode eval
```
