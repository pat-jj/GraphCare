export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

python graphcare.py --dataset mimic3 --task mortality --kg GPT-KG --batch_size 64 --hidden_dim 512 --epochs 100 --lr 1e-3 --weight_decay 1e-4 --dropout 0.5 --num_layers 3 --decay_rate 0.01 --patient_mode joint --edge_attn True --device 1