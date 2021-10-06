TEST_FLAG = True
TRAIN_FLAG = True
CACHING_FLAG = True
EPOCH_LEVEL_CACHING = False

CREATE_FIDELITY_CURVES = False
NUM_FIDELITY_CURVE_SAMPLES = 1

FIDELITY_OCCLUSION_RATES = [x / 20 for x in range(0, 21)]

OUTPUT_DIR = "../output"

dataset_dict = {'dataset': ["fever"]}

dataset_info = {
	'boolq': {
		"name": "Boolq",
		"data_dir": "",
		"train_path": "../csv/boolq/train.csv",
		"dev_path": "../csv/boolq/val.csv",
		"test_path": "../csv/boolq/test.csv",
		'classes': [False, True],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,		
	},
	'cose': {
		"name": "cose",
		"data_dir": "",
		"train_path": "../csv/cose/train.csv",
		"dev_path": "../csv/cose/val.csv",
		"test_path": "../csv/cose/test.csv",
		"classes": ['A','B','C','D','E'],
		# "classes": ["FALSE","TRUE"],
		# 'classes': [False, True],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,		
	},
	'scifact': {
		"name": "SCIFACT",
		"data_dir": "",
		"train_path": "../csv/scifact/train.csv",
		"dev_path": "../csv/scifact/val.csv",
		"test_path": "../csv/scifact/test.csv",
		"classes": ['REFUTES', 'SUPPORTS'],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,		
	},
	'multirc': {
		"name": "multirc",
		"data_dir": "",
		"train_path": "../csv/multirc/train.csv",
		"dev_path": "../csv/multirc/val.csv",
		"test_path": "../csv/multirc/test.csv",
		'classes': [False, True],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'fever': {
		"name": "FEVER",
		"data_dir": "",
		"train_path": "../csv/fever/train.csv",
		"dev_path": "../csv/fever/val.csv",
		"test_path": "../csv/fever/test.csv",
		'classes': ['REFUTES', 'SUPPORTS'],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'movies': {
		"name": "movies",
		"data_dir": "",
		"train_path": "../csv/movies/train.csv",
		"dev_path": "../csv/movies/val.csv",
		"test_path": "../csv/movies/test.csv",
		'classes': ['NEG', 'POS'],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'sst': {
		"name": "Stanford treebank",
		"data_dir": "",
		"train_path": "",
		"dev_path": "",
		"test_path": "",
		"classes": ['neg', 'pos'],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'sstred': {
		"name": "Stanford treebank Reduced",
		"data_dir": "",
		"train_path": "",
		"dev_path": "",
		"test_path": "",
		"classes": ['neg', 'pos'],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'moviesred': {
		"name": "movie reviews Reduced",
		"data_dir": "",
		"train_path": "",
		"dev_path": "",
		"test_path": "",
		'classes': ['NEG', 'POS'],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'multircred': {
		"name": "MultiRC Reduced",
		"data_dir": "",
		"train_path": "",
		"dev_path": "",
		"test_path": "",
		'classes': [False, True],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'feverred': {
		"name": "FEVER Reduced",
		"data_dir": "",
		"train_path": "",
		"dev_path": "",
		"test_path": "",
		'classes': ['REFUTES', 'SUPPORTS'],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'wikiattack': {
		"name": "Wikipedia personal attacks",
		"data_dir": "",
		"train_path": "",
		"dev_path": "",
		"test_path": "",
		'classes': [0, 1],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'wikismall': {
		"name": "Wikipedia personal attacks Small",
		"data_dir": "",
		"train_path": "",
		"dev_path": "",
		"test_path": "",
		'classes': [0, 1],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'wikismallred': {
		"name": "Wikipedia personal attacks Small Reduced",
		"data_dir": "",
		"train_path": "",
		"dev_path": "",
		"test_path": "",
		'classes': [0, 1],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'esnli': {
		"name": "E-SNLI",
		"data_dir": "",
		"train_path": "",
		"dev_path": "",
		"test_path": "",
		'classes': ['contradiction', 'entailment', 'neutral'],
		"batch_size": 512,
		"max_rows": None,
		"max_len": 512,
	},
	'esnlired': {
		"name": "E-SNLI Reduced",
		"data_dir": "",
		"train_path": "",
		"dev_path": "",
		"test_path": "",
		'classes': ['contradiction', 'entailment', 'neutral'],
		"batch_size": 128,
		"max_rows": None,
		"max_len": 512,
	},
}
