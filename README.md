# CS762 Project

This project explores RNN for predicting text choices.

## Explore neural networks

The first notebook is focused on understanding the basics
of neural networks, the operations the represent, and the 
influence of meta-parameters.

nb/basic-nn.ipynb

It hand implements a simple NN and explores how teaching it 
to count, i.e. add one to the previous number.

## Explore gensim

This notebook follows the [gensim getting started tutorials
for gensim](https://radimrehurek.com/gensim/tutorial.html)

nb/gensim-ex.ipynb

## Wikipedia topic modeling

The project works with the Wikipedia corpus to build 
an LDA topic model.

### Running the notebooks

Because these notebooks work with larger models they favor
multi-process solutions for distributing work.  They can be
loaded just like the notebooks in class but should reserve
a more significant chunk of resources. This is easiest to
do by reserving a whole compute node with the -N1 and
--exclusive flags.  Given the resource need it's easiest
to use the pascalnodes with 28 cores. We request a GPU for 
good measure incase models avail themselves to GPU use:

```
srun -N1 --exclusive --time=12:00:00 --partition=pascalnodes \
     --gres=gpu --job-name=rnn --pty --exclude='c[0097-0100]' \
     /bin/bash

```


### Stage pre-trained models

The wikipidia corpus is 15Gigabyte data set that requires
significant computational resources during training. Using
the project's pre-trained models.  This is easily done
on cheaha by copying the data directory into to the top-
level project directory:
```
cp -r /home/jpr/projects/cs762/project/data .
```

## Wikipedia topic modeling: pass one

The gensim site outlines how to work with the wikipedia
data as a corpus for building topic models.  The goal
is to use these topic models to classify documents of 
interest.

### Build Wikipedia dictionary

These notebooks rely on a dictionary and document-word
matrix data set built from the Wikipedia article corpus.

This section can be skipped if the data is pre-staged
using as described above.

Building these data sets takes many hours, so it's 
to simply copy the project data directory as documented
above so the notebooks can efficiently use the resulting
models.

Building the wikipedia dictionary and tfidf document-word 
matrix can be done out of band with the wiki utility script.

Download the [2018-11-20 archive of the wikipedia data set](http://dumps.wikimedia.your.org/enwiki/20181120/)
using the your.org mirror to provide fast downloads.

```
cd data
if [ ! -f enwiki-20181120-pages-articles.xml.bz2 ]
then
   wget http://dumps.wikimedia.your.org/enwiki/20181120/enwiki-20181120-pages-articles.xml.bz2
fi
```

Run the dictionary and tfidf matrix generation utility. This utility
will use all cores available to process the documents more efficiently.
Using one of the 28-core pascal nodes on cheaha provides the 
best peformance but will still take almost six hours.  Unfortunately,
the code is note GPU capable.

```
srun -N1 --exclusive --partition=pascal nodes \
  python -m gensim.scripts.make_wiki \
     enwiki-20181120-pages-articles.xml.bz2 \
     `pwd`/wikipedia_ex/wiki
```

### Build the LDA model

This notebook uses the dictionary to train an LDA model.
It follows similar logic to the [example gensim tutorial
for exploring the wikipedia data set](https://radimrehurek.com/gensim/wiki.html)
but favors the multi-process LDA model for efficiency.


nb/gensim-wikipedia-experiments.ipynb

### Test LDA model

This notebook demonstrates loading the pre-trained model
and analyzing a document for its topic similarity.

nb/gensim-wiki-models.ipynb
