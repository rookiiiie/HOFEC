        nohup\
        python traineval.py --HO3D_root E:/localPy/HOFEC/data/ho3d_v2 \
        --host_folder  host_folder/ho3d \
        --dex_ycb_root E:/localPy/HOFEC/data/DEX_YCB \
        --epochs 40 \
        --inp_res 256 \
        --lr 1e-4 \
        --train_batch 58 \
        --mano_lambda_regulshape 0 \
        --mano_lambda_regulpose  0 \
        --lr_decay_gamma 0.7 \
        --lr_decay_step 8 \
        --test_batch 128 \
        --use_ho3d \
        > train_check_ho3d.log 2>&1 &

        # CUDA_VISIBLE_DEVICES=0
        #        CUDA_VISIBLE_DEVICES=0,1,3,4\
        
