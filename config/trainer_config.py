training_args_config = {
	"overwrite_output_dir": True,
	"do_train": True,
	"do_eval": True,
	"do_predict": True,
	"evaluation_strategy": "steps",
	# "per_device_train_batch_size": 16,
	"per_device_eval_batch_size": 16,
	# "learning_rate": 2e-5,
	"logging_steps": 500,
	# "num_train_epochs": 10,
	# "warmup_steps": 50,
	"logging_dir": "",
}
tunable_training_args = {

	# "learning_rate": [5e-6, 1e-5, 2e-5],
	"learning_rate": [5e-6, 1e-5, 2e-5, 5e-5],
	# "learning_rate": [1e-5, 2e-5, 3e-5, 4e-5, 8e-5, 1e-4, 7e-4, 1e-3]
	# "learning_rate": [1e-5, 2e-5],
	# "learning_rate": [2e-5],
	# "learning_rate": [5e-4, 1e-3, 2e-3],
	# "learning_rate": [5e-10],
	# "learning_rate": [1e-3],
	# "num_train_epochs": [10],
	# "num_train_epochs": [10],
	# "pad_packing": True,
	# "weight_decay": 1e-6
}


# try packed padded
# do packing as a hyper-parameter set this as true and false and see if it makes a difference
