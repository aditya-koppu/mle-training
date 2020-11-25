
Installing the environment using conda:
	conda env create -f env_me.yml
Refer to conda documentation incase of any questions: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

Verifying if all the necessary modules are installed (from proj folder)(cd ...../proj/)
	python 
	import scripts as s
	s.check()

If "All necessary modules installed" is displayed then please proceed. Else install the necessary(printed out) modules using conda or pip.

Note: All the code is taken from the previous file. Some of the redundant code has been commented out. Nothing new was written. Just chopped up 
	  to create the following train, evaluate and test scripts. These scripts can be run from command line. They are NOT made to take in arguments.
	  They print out if it was sucessful. Statistics are printed out in log.

Installing as library
	pip install -e .

Command to Train models (refer readme to know the models being trained)
	import scripts.train_me as stn
	stn.train_all()


Command to score models 
	import scripts.score_me as ssm
	ssm.score()

Command to run final model on test data
	import scripts.test_me as stm
	stm.final_pred_res()

