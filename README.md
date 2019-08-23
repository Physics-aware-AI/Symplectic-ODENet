for experiment-single task there are three different models

```bash
python experiment-single-force/train.py --verbose --num_points 4 --gpu 3 
python experiment-single-force/train.py --verbose --num_points 4 --gpu 3 --structure
python experiment-single-force/train.py --verbose --num_points 4 --gpu 3 --baseline

```

for experiment-single-embed task, there are four different models

```bash
python experiment-single-embed/train.py --verbose --num_points 4 --gpu 3 
python experiment-single-embed/train.py --verbose --num_points 4 --gpu 3 --structure
python experiment-single-embed/train.py --verbose --num_points 4 --gpu 3 --baseline
python experiment-single-embed/train.py --verbose --num_points 4 --gpu 3 --naive

```

