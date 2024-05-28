# Naive LSTM
We train a navie LSTM to perform the word deletion task (see Methods). The LSTM is based on the Match-LSTM, and the code is forked from [this repo](https://github.com/laddie132/Match-LSTM).


## Result in Paper
To replicate the results in our papers, please download the glove embedding ([glove.840B.300d.zip](https://nlp.stanford.edu/data/glove.840B.300d.zip)) and the SGNS chiese embedding ([sgns.context.character.char1-4.bz2](https://pan.baidu.com/s/1hJKTAz6PwS7wmz9wQgmYeg)), and move them to the `data/` directory.

Then run the script:

```bash
bash run.sh
```

## Run Your Own LSTM

The configuration of the LSTM and the experiment can be switched in the `config/` directory.