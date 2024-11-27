
device=0

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale
        save_dir=C:/Users/Deept/OneDrive/Desktop/Github/MedVMAD/checkpoint/
        CUDA_VISIBLE_DEVICES=${device} python test_one_example.py \
        --image_path C:/Users/Deept/OneDrive/Desktop/Github/MedVMAD/data/Brain_AD/test/Ungood/img/01597_80.png \
        --checkpoint_path ${save_dir}epoch_15.pth \
         --features_list 6 12 18 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done

