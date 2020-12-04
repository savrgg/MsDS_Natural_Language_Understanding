# virtual env =====
# cd 
# cd virtualenvs/nlu
# source ./bin/activate
# cd ~/Repositories/RP_edinburgh_master/NLU/Assignment2/src/

# Estimate ========
# sys.argv[1] = estimate
python rnn.py estimate ../data/

# Train ===========
# sys.argv[1] = train
# sys.argv[2] = number of hidden units
# sys.argv[3] = number of look-back layers
# sys.argv[4] = learning rate
python rnn.py train ../data/ 50 2 0.5

# Generate ========
# sys.argv[1] = generate
# sys.argv[2] = path to data
# sys.argv[3] = path to files U, V, W
# sys.argv[4] = limit of the sentence length
python rnn.py generate ../data/ ./ 30
