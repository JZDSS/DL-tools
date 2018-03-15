python3 train_ssd.py --batch_size 16 --max_steps 80000 --learning_rate 1e-3 --trainable_scope l2_norm,conv8,conv9,conv10,conv11
python3 train_ssd.py --batch_size 8 --max_steps 160000 --start_step 80000 --learning_rate 1e-5

