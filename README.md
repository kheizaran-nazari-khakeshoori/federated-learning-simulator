# Federated Learning Simulator

## Installation

```bash
pip install -r requirements.txt
sudo apt install python3-tk
```

## Usage

```bash
python3 app.py
# or make run
```

## Performance Tips

- alpha 10 + lr 0.05 + C 0.5 = smooth 30fps
- Use Fast preset for demo, Accurate for final

## Docker

```bash
docker build -t fl-sim .
docker run -e DISPLAY $DISPLAY fl-sim
```
