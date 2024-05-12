        nohup\
        python traineval.py --HO3D_root E:/localPy/HOFEC/data/ho3d_v2 \
        --host_folder  host_folder/ho3d \
        --dex_ycb_root /data1/zhifeng/dex-ycb \
        --epochs 70 \
        --inp_res 256 \
        --lr 1e-4 \
        --train_batch 64 \
        --mano_lambda_regulshape 0.2 \
        --mano_lambda_regulpose  0.2 \
        --lr_decay_gamma 0.7 \
        --lr_decay_step 10 \
        --test_batch 128 \
        --use_ho3d \
        --evaluate \
        --resume E:/localPy/HOFEC/host_folder/ho3d/checkpoint_45.pth.tar \
        > train_check_ho3d_test.log 2>&1 

        #        CUDA_VISIBLE_DEVICES=0,1,3,4\
        # --resume E:/localPy/HOFEC/host_folder/ho3d/checkpoint_70.pth.tar \
