

python examples/08_tmaze.py --memory_size 128 --memory_layers 2 --horizon 10000 --timesteps_per_epoch 10100  \
    --batch_size 18 --grads_per_epoch 600 --dset_max_size 5000 --run_name tmaze_seed$SEED  --buffer_dir ./buffer_dir/  \
    --expt_seed $SEED --parallel_actors 24

python examples/08_tmaze.py --memory_size 128 --memory_layers 2 --horizon 10000 --timesteps_per_epoch 10100  \
    --batch_size 18 --grads_per_epoch 600 --dset_max_size 5000 --run_name tmaze_ac_seed$SEED  --buffer_dir ./buffer_dir/ \
    --tformer_type "autocomp_tformer" --summary_length 4 --segment_length 256  --expt_seed $SEED --parallel_actors 24