export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python graphcare.py --dataset mimic3 --task mortality --kg GPT-KG --batch_size 4 --hidden_dim 512 --epochs 100 --lr 1e-5 --weight_decay 1e-5 --dropout 0.5 --num_layers 1 --decay_rate 0.01 --freeze_emb False --patient_mode joint --edge_attn True --attn_init False --device 1