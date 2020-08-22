# Semantic Search using Sentence BERT
With the introduction of Natural Language processing models, it would seem that keyword matching type of search does not seem to do the trick anymore hence why semantics would need to be introduced into search engines. Semantic search means searching with meaning, instead of searching for keywords, it brings inference and contextual meaning to find related entries within the results. With the emergence of the powerful BERT model in 2018 by Google, such semantic derivatives are easier to get. Words are embedded and represented as numeric vectors and then use cosine similarity function to compute how similar the entries are.

Sentence BERT, a flavour of BERT which is more suitable for embedding sentences as training a BERT model from scratch can take days to complete. More information on this model can be found in this paper:

Ref: ACL 2019 paper entitled “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks” by Nils Reimers and Iryna Gurevych.
