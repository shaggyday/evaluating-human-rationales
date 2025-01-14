import logging
import sys
sys.path.append('..')

from util.choose_gpu import choose_and_set_available_gpus
choose_and_set_available_gpus()
# sys.exit(0)
from transformers import Trainer, TrainingArguments, EvalPrediction, PretrainedConfig
from typing import Callable, Dict
import sklearn.metrics as mt
import numpy as np

from config import data_config as config
from dataset.dataset import prepare_data
from config.trainer_config import training_args_config, tunable_training_args
from config.model_config import model_dict, model_info
from util.param_combo import get_param_combos
from train_eval.feature_caching import get_and_save_features
from dataset.dataset import create_test_dataloader, create_dataloader
from train_eval.create_fidelity_curves import create_fidelity_curves
from train_eval.eval_pytorch import eval_fn
from util.saving_utils import copy_features

import os
import torch
import json

# logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler())
# logging.basicConfig(
# 	format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
# 	datefmt="%m/%d/%Y %H:%M:%S",
# 	level="CRITICAL",
# 	# level=logging.DEBUG,
# )

import warnings

warnings.filterwarnings("ignore")

def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
	def compute_metrics_fn(p: EvalPrediction):
		preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
		# if output_mode == "classification":
		# 	preds = np.argmax(preds, axis=1)
		# else:  # regression
		# 	preds = np.squeeze(preds)
		preds = np.argmax(preds, axis=1)
		
		# score = mt.precision(p.label_ids, preds)
		# score = mt.recall(p.label_ids, preds)
		# score = mt.f1_score(p.label_ids, preds)
		# score = mt.roc_auc_score(p.label_ids, preds)
		score = mt.accuracy_score(p.label_ids, preds)
		return {"acc": score}

	return compute_metrics_fn

OUTPUT_DIR = config.OUTPUT_DIR

DATASET_DICT = config.dataset_dict
DATASET_INFO = config.dataset_info
# TRAINING_PARAM_DICT = config.training_param_dict

CACHING_FLAG = config.CACHING_FLAG
EPOCH_LEVEL_CACHING = config.EPOCH_LEVEL_CACHING
TRAIN_FLAG = config.TRAIN_FLAG
TEST_FLAG = config.TEST_FLAG

CREATE_FIDELITY_CURVES = config.CREATE_FIDELITY_CURVES
NUM_FIDELITY_CURVE_SAMPLES = config.NUM_FIDELITY_CURVE_SAMPLES
FIDELITY_OCCLUSION_RATES = config.FIDELITY_OCCLUSION_RATES


larger_models = ["longformer", "roberta-large"]

"""
for all model types:
	for all param_combos (dataset, model parameteres, training parameters)
		train
		eval
		eval_fidelity
"""
# from transformers import EvalPrediction

if __name__ == "__main__":
	for model_name in model_dict['model']:
		print(f"===============Training on Model: {model_name}===================")
		tunable_model_args = model_info[model_name]["tunable_model_args"]
		param_combos = get_param_combos([DATASET_DICT, tunable_model_args, tunable_training_args])
		dataset_prediction_caching_info = {}
		for dataset in DATASET_DICT['dataset']:
			prediction_caching_info = {"best_dev_acc": 0.0,
									   "path": [os.path.join(OUTPUT_DIR, os.path.join(model_name, dataset))]}
			dataset_prediction_caching_info[dataset] = prediction_caching_info

		for param_combo in param_combos:
			print("====================================    param_combo:    ========================================")
			print(param_combo)
			dataset = DATASET_INFO[param_combo["params"][0]["dataset"]]
			model_dict = model_info[model_name]
			tunable_model_args = param_combo["params"][1]
			tunable_training_args = param_combo["params"][2]
			output_dir = os.path.join(OUTPUT_DIR, os.path.join(model_name, param_combo["name"]))
			best_model_save_path = os.path.join(OUTPUT_DIR,
												os.path.join(model_name, param_combo["params"][0]["dataset"]))
			# Model Class
			model_config = PretrainedConfig(
				max_length=dataset["max_len"],
				num_labels=len(dataset["classes"]),
				**tunable_model_args)
			
			if TEST_FLAG:
				warmup_steps = 0
				num_train_epochs = 1
			elif model_name == "lstm":
				num_train_epochs = 5
				warmup_steps = 0
			else:
				num_train_epochs = 10
				warmup_steps = 50


			candidate_model = model_dict["class"](config=model_config)
			# Get the data and create Dataset objects
			if TRAIN_FLAG:
				train_dataset, eval_dataset = prepare_data(
					model=candidate_model,
					return_dataset=True,
					**dataset)

				training_args_config["per_device_train_batch_size"] = dataset["batch_size"]
				if model_name in larger_models:
					training_args_config["per_device_train_batch_size"] = int(training_args_config["per_device_train_batch_size"]/2)

				# Save every epoch checkpoint which could be used for analysis later
				save_steps = len(train_dataset) // training_args_config['per_device_train_batch_size']
				
				training_args = TrainingArguments(
					output_dir=output_dir,
					save_strategy="no",
					# save_steps=save_steps,
					num_train_epochs=num_train_epochs,
					warmup_steps=warmup_steps,
					**training_args_config,
					**tunable_training_args)

				trainer = Trainer(
					model=candidate_model,
					args=training_args,
					train_dataset=train_dataset,
					eval_dataset=eval_dataset,
					compute_metrics=build_compute_metrics_fn(),
					# tokenizer=candidate_model.tokenizer
				)

				# Training, Evaluating and Saving
				if config.TRAIN_FLAG:
					print(
						f"===============Training on Dataset: {dataset['name']} and param combo: {param_combo['name']}===================")
					trainer.train()
					trainer.save_model(output_dir=output_dir)
					# Evaluate all epochs and save the best one


			# Caching features for analysis
			if CACHING_FLAG:
################ loading trained model ###########################

				LOAD_DIR = os.path.join(OUTPUT_DIR, os.path.join(model_name, param_combo["name"]))
				# LOAD_DIR = best_model_save_path
				# LOAD_DIR_LIST = [LOAD_DIR]
				LOAD_DIR_LIST = []
				# LOAD_DIR_LIST.append(best_model_save_path)
				test_dataloader = create_test_dataloader(
					model=candidate_model,
					filepath=dataset["test_path"],
					classes=dataset["classes"],
					batch_size=training_args_config["per_device_eval_batch_size"],
					name=dataset['name']
				)

				if EPOCH_LEVEL_CACHING:
					print("???")
					LOAD_DIR_LIST = LOAD_DIR_LIST + [
						os.path.join(LOAD_DIR, name) for name in os.listdir(LOAD_DIR) if "checkpoint-" in name
					]

					# Save features for epoch zero
					cache_model = model_dict["class"](config=model_config)
					get_and_save_features(
						test_dataloader=test_dataloader,
						model=cache_model,
						tokenizer=cache_model.tokenizer,
						save_dir=os.path.join(LOAD_DIR, "epoch-0"),
					)
				else:
					LOAD_DIR_LIST = [LOAD_DIR]

				if CREATE_FIDELITY_CURVES:
					print(f'Creating fidelity curves with {NUM_FIDELITY_CURVE_SAMPLES} sample(s) each for occlusion rates: \n{FIDELITY_OCCLUSION_RATES}')

					model_load_path = os.path.join(LOAD_DIR, 'pytorch_model.bin')
					cache_model = model_dict["class"](config=model_config)
					cache_model.load_state_dict(torch.load(model_load_path))

					create_fidelity_curves(
						model=cache_model,
						dataset_path=dataset["test_path"],
						dataset_classes=dataset["classes"],
						batch_size=training_args_config["per_device_eval_batch_size"],
						output_dir=os.path.join(LOAD_DIR, 'fidelity_curves'),
						num_samples=NUM_FIDELITY_CURVE_SAMPLES,
						occlusion_rates=FIDELITY_OCCLUSION_RATES
					)
					
				for load_path in LOAD_DIR_LIST:
##################  still loading trained model ##################
					print(f"===============Feature caching on Dataset: {dataset['name']} and"
						f" param combo: {param_combo['name']}, load path {load_path} ===================")

					# cache_model = RobertaClassifier.from_pretrained(load_path)
					model_load_path = os.path.join(load_path, 'pytorch_model.bin')
					with open(os.path.join(load_path, 'config.json'), 'r')as f:
						saved_config = json.load(f)
					saved_config = PretrainedConfig(
						num_labels=len(dataset["classes"]),
						**saved_config
					)
##################  finally loading trained model ##################
					cache_model = model_dict["class"](config=saved_config)
					cache_model.load_state_dict(torch.load(model_load_path))

##################  running on test dataset and saving features ##################
					# look at the output _dir
					get_and_save_features(
						test_dataloader=test_dataloader,
						model=cache_model,
						tokenizer=cache_model.tokenizer,
						save_dir=load_path,
					)

##################  evaluating on dev dataset???? ##################
					# Get the epoch with best dev acc
					# dev_dataloader = create_dataloader(cache_model, dataset["classes"], dataset["dev_path"], dataset["batch_size"])
					dev_acc, _ = eval_fn(cache_model, test_dataloader) #, 5)

					print(f"Dataset: {param_combo['params'][0]['dataset']}| Eval Acc: {dev_acc}")
					if dev_acc > dataset_prediction_caching_info[param_combo["params"][0]["dataset"]]["best_dev_acc"]: #rhs == 0
						print(f"NEW HIGHEST RECORD!!!!!!!!!!!!!!!!!!!!!!!!")
						print(f"Path: {model_load_path}")
						print(f"load_path: {load_path}| best+_model_save_path: {load_path}")
						# copy_features(load_dir=load_path, output_dir=best_model_save_path)
						dataset_prediction_caching_info[param_combo["params"][0]["dataset"]]["best_dev_acc"] = dev_acc

	print("Done!")


