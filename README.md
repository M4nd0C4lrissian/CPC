# CPC

We will point you to the high level flow for behavior, but changing file pathing will have to be up to interested parties. Note that we only include the preset behavior to generate CPC and FNPC, and do not consider the NN and FN-NN methods, though they could easily be done by changing the user_correlation matrix used in the files.

The repository is organized in the following manner:

bert_embedding_generation contains ipynb files to generate bert embeddings:

- data_generation.ipynb
- BertEmbeddings.ipynb

NOTE: the dataset used comes from prior work, and a link to download it can be found here: https://github.com/IIT-ML/WWW21-FilterBubble. User bias matrices can also be generated from user_generation.py, which comes from this prior work as well.

Then we have the Disentangler training in disentangler:

- training.py
- and the custom dataloader

In make_CPC we simulate user choice and split up our test data into the UI matrix and the holdout ratings. 

- user_choice.py
- split_test_data.py

We then generate landmarks and CPC:

- create_landmarks.py
- baseline_correlation_matrix.py 

We then train our GCFs in GCF_training:

- train_GCF.py

We then generate recommendations using CF_model_evaluation.py (this file is quite bloated, but it will create the necessary files)

We also include the code to generate WD and recommendation entropy in evaluate.