#!/bin/bash
declare -a test_cases=("NoPreTrain" "QuarterPreTrain" "HalfPreTrain" "ThreeQuarterPreTrain")
declare -a methods=("NoModel" "ICM" "RND" "NGU" "NovelD" ) # "DEIR" "AEGIS"
declare -a arr=("ninja" "climber" "jumper")

for group_name in "${test_cases[@]}"; do
  for env in "${arr[@]}"; do
    for int_rew_source in "${methods[@]}"; do
      for seed in 0 1 2; do
        # Set total training steps based on the environment
        if [ "$env" == "ninja" ]; then
          total_steps=100_000_000
        else 
          total_steps=200_000_000
        fi

        # Logging explored states options
        # 0 - Not to log
        # 1 - Log both episodic and lifelong states
        # 2 - Log episodic visited states only
        log_explored_states=1
        
        # Default hyperparameters for intrinsic rewards
        int_rew_momentum=0.9
        rnd_err_norm=1
        int_rew_coef=5e-2
        
        # Adjust hyperparameters based on the intrinsic reward method
        if [ "$int_rew_source" == "NGU" ]; then
          int_rew_coef=3e-4
          int_rew_momentum=0.0
        elif [ "$int_rew_source" == "NovelD" ]; then
          int_rew_coef=3e-2
          rnd_err_norm=0
        elif [ "$int_rew_source" == "RND" ]; then
          int_rew_coef=1e-2
          rnd_err_norm=0
        elif [ "$int_rew_source" == "ICM" ]; then
          int_rew_coef=1e-4
          rnd_err_norm=0
        fi

        # Set pretraining percentage based on the group name
        if [ "$group_name" == "ThreeQuarterPreTrain" ]; then
          pretrain_percentage=0.75
        elif [ "$group_name" == "HalfPreTrain" ]; then
          pretrain_percentage=0.5
        elif [ "$group_name" == "QuarterPreTrain" ]; then
          pretrain_percentage=0.25
        else
          pretrain_percentage=0.0
        fi
        
        PYTHONPATH=./ python3 src/train.py \
        --group_name=$group_name \
        --run_id=$seed \
        --total_steps=$total_steps \
        --pretrain_percentage=$pretrain_percentage \
        --int_rew_source=$int_rew_source \
        --env_source=procgen \
        --game_name=$env \
        --num_processes=32 \
        --n_steps=256 \
        --batch_size=2048 \
        --n_epochs=3 \
        --model_n_epochs=3 \
        --learning_rate=1e-4 \
        --model_learning_rate=1e-4 \
        --latents_dim=256 \
        --features_dim=256 \
        --model_features_dim=64 \
        --policy_cnn_type=2 \
        --policy_cnn_norm=LayerNorm \
        --policy_mlp_norm=NoNorm \
        --model_cnn_type=1 \
        --model_cnn_norm=LayerNorm \
        --model_mlp_norm=NoNorm \
        --adv_norm=0 \
        --int_rew_coef=$int_rew_coef \
        --int_rew_momentum=$int_rew_momentum \
        --rnd_err_norm=$rnd_err_norm
        
        # redirect terminal output to null
      done
    done
  done
done
echo "All experiments done!"