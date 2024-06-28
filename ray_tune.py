from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
import model
import config as cfg
import os, datetime

ROOT_PATH = os.getcwd()

def tuning_hyper(m_config:dict):
    g_config = cfg.get_general_config()

    data_dirname = g_config['data_src']['dirname']
    checkpoint_dirname = g_config['model']['checkpoint']['dirname']
    
    corpus_file = f"{ROOT_PATH}/{data_dirname}/{g_config['data_src']['corpus_file_name']}"
    sp_model_file = f"{ROOT_PATH}/{data_dirname}/{g_config['data_src']['sp_model_file']}"
    tmp_dir = f"{ROOT_PATH}/tmp"
    checkpoint_dir = f"{ROOT_PATH}/{checkpoint_dirname}"

    gpt_model = model.train(m_config,
                            g_config,
                            corpus_file=corpus_file,                            
                            sp_model_file=sp_model_file,
                            tmp_dir=tmp_dir,
                            checkpoint_dir=checkpoint_dir,
                            epochs=m_config['epochs'],
                            tuning_mode=True)
    print("Training has finished.")

def trial_str_creator(trial):
    str_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    trialname = f"{trial.trainable_name}_{trial.trial_id}_{str_now}"
    return trialname


def main():
    g_config = cfg.get_general_config()

    search_space = {
        "epochs": 20,
        "train_batch_size": 32,
        "valid_batch_size": 32,
        "sequence_len": tune.choice([250, 512]),
        "num_decoders": 12,
        "embed_dim": tune.choice([384, 576, 768]),
        "num_heads": 12,
        "ff_dim": tune.choice([433, 541]),
        "dropout_prob": 0.2,
        "activation_name": 'gelu',
        "norm_placing": 'post',
        "learning_rate": tune.choice([1e-3, 1e-2, 1e-1])
    }

    # _trainable = tune.with_resources(tuning_hyper, {"cpu": 0})
    
    tuner = tune.Tuner(
        tuning_hyper,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            trial_name_creator=trial_str_creator,
            trial_dirname_creator=trial_str_creator,
            mode="min",
            metric="valid_cost",
            num_samples=5
        )
        # run_config=train.RunConfig(
        #     stop={"training_iteration": 10}
        # )
    )
    result = tuner.fit()
    config_dict = result.get_best_result(mode="min", metric="valid_cost").config

    best_checkpoint = result.get_best_result(mode="min", metric="valid_cost").get_best_checkpoint(mode="min", metric="valid_cost")
    if best_checkpoint:
        with best_checkpoint.as_directory() as chkp_dir:
            gpt_model = model.load_model_state_dict(config_dict, os.path.join(chkp_dir, g_config['model']['tuning']['checkpoint_basename']))
            
            cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            export_dirname = g_config['model']['final']['dirname']
            model_state_file = f"{ROOT_PATH}/{export_dirname}/{g_config['model']['final']['state_file_basename']}_{cur_time}.pt"

            saved_as = model.save_model_state_dict(gpt_model, model_state_file)
            print(f"Best model state saved as: {saved_as}")

if __name__ == "__main__":
    main()