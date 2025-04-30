python train.py --model_type=est_pose --exp_name=task --device=cuda:0

python test.py --checkpoint=exps/task1/checkpoint/checkpoint_10000.pth --mode=val

python train.py --model_type=est_coord --exp_name=task2 --device=cuda:0

python test.py --checkpoint=exps/task2_2/checkpoint/checkpoint_20000.pth --mode=val

python eval.py --checkpoint=exps/task2_2/checkpoint/checkpoint_20000.pth --mode=val --vis=1 --headless=0

