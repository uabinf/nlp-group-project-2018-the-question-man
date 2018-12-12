# CS762 Project

<!-- TOC depthFrom:2 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Overview](#overview)
- [Wikipedia topic modeling](#wikipedia-topic-modeling)
	- [Running the notebooks](#running-the-notebooks)
	- [Stage pre-trained models](#stage-pre-trained-models)
	- [Explore LDA model](#explore-lda-topic-model)
	- [Build Wikipedia dictionary](#build-wikipedia-dictionary)
	- [Build the LDA model](#build-the-lda-model)
	- [Filter Wikipedia data set](#filter-wikipedia-data-set)
	  - [Getting started](#getting-started)
- [Explore neural networks](#explore-neural-networks)
- [Explore gensim](#explore-gensim)

<!-- /TOC -->
## Overview

This project explores the generation topic models from Wikipedia articles.

This is a complete dump of all Wikipedia pages in the English language wikipedia.
Each page in the dump corresponds to a single page on the human visible Wikipedia web site http://en.wikiwiki.org.
A page is the information content associated with the subject of the page as defined by the page title.
The expected content and length of a page can vary significantly depending on the subject and expected documentation conventions defined by a community of volunteer editors.
Biographies of historical figures, summaries of well-known locales, or other popular topics will tend to have comprehensive coverage.
Some pages originate from a public domain release of Encylopedia Britannica from the eary 20th century, potentially inheriting the editorial conventions and perspectives of that period.
Other pages are freshly created pages on as yet undocumented topics of interest to a small number of people.

A page is stored in a structured XML object that includes the namespace, title (subject) of the page, metadata on the current revision of the page, and wikitext of the current article.

Wikipedia is divided into a number of namespaces for different page types.
The article namespace contains the infomation typically associated with Wikipedia and was the sole focus of this project.
Other namespaces include pages used by editors to discuss article content and templates to facilitate common operations.
These namespaces were ignored during archive processing.

Wikitext is a semi-structured markup language that serves a variety of, often conflicting, needs.
There is the natural language text that contains the information conveyed in the article.
There is text markup meant to identify visual representation for display of the text in HTML within a users web browser.
This includes markup for bold or italic fonts, headings on the page or other HTML documentation conventions, like links to other pages (articles in Wikipedia or websites on the internet).
Additional markup conventions have been developed to provide structured representation of common types of information, like demographics for different geographic or political boundaries.
Many of these conventions are implemented as template references with special start and stop symbols.
Parsing wiki text to separate markup from the natural language text that makes up the information content of the page is a complex endevor.
Parsing wiki text involves many text processing tasks beyond tokenizing words from the running text and  normalizing the word formats.

For this project we focused on building topic models using Latent Dirichlet Analysis (LDA) as implemented in the gensim toolkit.
The LDA was applied to the extracted the text field (wikitext) from the XML page object.
The wikitext was processed to isolate the English language article text (information content) from common wikitext markup.

Three approaches to corpus construction were explored in this project.
Both approaches sought to extract the information content from the page's wikitext.
The first approach used a Wikipedia processing utility (make_wiki) distributed with the gensim toolkit.
This utility processes each page object in a given compressed archive composed of concatenated XML page objects.
There are two passes over the input data set, the first extracts the plain text (information text) from the articles and builds a vocabulary for the corpus.
The vocabulary is constructed from the 100,000 most frequent words that appear in no more than 10% of the articles.
The second pass builds a TF-IDF matrix represenation of the articles.
The make_wiki utility uses a collection of regular expresssions to extract the plain text from wikitext.

The full Wikipedia archive is 14Gigabytes of compressed data.
The represents a formidable amount of data and indeed took considerable time to ingest into the gensim sparse document vector model.
The initial processing of the full archive into a vocabulary and associated sparse document matrix took approximately six hours with 28-cores.
This resulted in a vocabulary of 100,000 words derived from 4.5 million articles.
This 4.5 million by 100,000 matrix contained 720 million non-zero entries.

Given the size of this data set and the time needed to process, two approaches were used to work with a reduced data set.
These two approaches represent the second and third corpus construction explorations referenced above.

The first attempt at a reduced development corpus was based on the discovery on a set of Rust-language utilities designed to parse the archive and wikitext.
These utilities are recommended by the Wikimedia project which produces the Wikipedia archive.
The developers for this library spend considerable effort producing a wikitext parser that produces highly structured tree of objects meant to emmulate the processing of an actual Wikipedia page.
They argue against simplistic regular expression parsers like those used in the make_wiki corpus generator from gensim.
Indeed, their library produced very capable parsed objects which contained "text" fields we could directly use as the source for a plain text article extraction.
We built an extraction utility that parsed the wikitext fields of pages in the full archive and produced plain text summaries of the first 1000 characters (approximately 200 words) of selected articles.
The utility further reduced the data set by selecting only every 1000th article from the archive.
This results in aproximately 4,500 articles of no more that 1000 characters.
This corpus promises to provide a very useful data set for futher development.
Unfortunately time constraints of the project prevented comparison of the quality of this archive subset to the reults of the full archive, in both model construction performance and model quality.

The third approach to a reduced corpus appropriate for development was based on focused extraction of Wikipedia articles.
Wikipedia supports per-acticle XML page dumps.
This API can be also be used to request multiple pages in a single archive that matches the format of the full dump.
This makes the resulting article dump useful as input to either of the corpus constructions approaches detailed above.
The collection of articles was generated from an dump of the archived browser requests of Wikipedia page views of one of the project members (JPR).
This collection dates back to February 2012 and represents about 1600 distinct wiki articles.
The page titles were extracted from the Firefox browser bookmarks table and pasted into the export form on Wikipedia.
This resulted in a match to approaximately 1000 Wikipedia pages exported to the XML dump format.
Due to time constraints, no effort was made to analyze the disparity of requested and provided articles.
This focused dump was then used to compare results of the topic distribution of a test document for the two models: the complete Wikipedia archive and the personal Wikipedia archive.

## Wikipedia topic modeling

The project works with the Wikipedia corpus to build
an LDA topic model.

The gensim site outlines how to work with the wikipedia
data as a corpus for building topic models.  The goal
is to use these topic models to classify documents of
interest.

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

### Explore LDA topic model

This notebook demonstrates loading the pre-trained model
and analyzing a document for its topic similarity.

nb/gensim-wiki-models.ipynb

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

### Filter Wikipedia data set

The wikipedia data set is too big to support easy iteration over building
the LDA model (each run takes hours).  There is evidence in early
results that suggests the text is dirty.

We can use existing filter tools to subsample the data set and clean the articles.

https://github.com/jprorama/parse_mediawiki_dump
https://github.com/jprorama/parse_wiki_text

#### Getting started

We need Rust.

Install rust:

```
curl https://sh.rustup.rs -sSf | sh`
```

Source the Rust install:

```
source $HOME/.cargo/env
```

Get my fork of the filter project:

```
git clone git@github.com:jprorama/parse_mediawiki_dump.git`
cd parse_mediawiki_dump
git checkout expanded-examples
cargo build --all-targets
```

Set up your environment to reference the example builds from within the
rust build tree:
```
PATH=`pwd`/target/debug/examples:$PATH
```

Explore the format of wikipedia dump and type of articles. Go to the data
directory of the class project `data` directory.  Look at some of the
output from the dump (interrupt the run when you get some data).

```
parse_dump enwiki-20181120-pages-articles.xml.bz2
article_type enwiki-20181120-pages-articles.xml.bz2
```

## Explore neural networks

An original aim for this project was to use RNNs for constructing Wikipedia article recommendations
based on thier relevance to web pages in a users browser history.
The idea was derive a sequence model, either from visited web site names or their topic coverage, and
recommend pages/topics that could follow and recommend Wikipedia pages that relate to those topics.
The motivatation was to provide information that could assist in the learning activities of
the user visiting these pages.

This aim remained unexplored due to the challenges confronted when trying to
build topic models from Wikipedia articles.

An initial attempt was made to better understand the function of Neural Network models
in general.
The first notebook is focused on understanding the basics
of neural networks, the operations the represent, and the
influence of meta-parameters.

nb/basic-nn.ipynb

It hand implements a simple NN and explores teaching it
to count, i.e. add one to the previous number.

## Explore gensim

This notebook follows the [gensim getting started tutorials
for gensim](https://radimrehurek.com/gensim/tutorial.html)

nb/gensim-ex.ipynb
