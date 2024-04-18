# Language of Music

The objective of Language of Music is to identify if lyrics of music have specific recurring themes to them, and to identify if they can be used to identify common splits in themes across different variables, such as genre. By being able to identify and analyze recurring lyrical themes, it may be possible to utilize topic models in a variety of interesting ways, within the realm of music. Here is just a few of the possible use cases:

- Music genre classification
- creative lyrical generation
- trend analysis
- artist signature identification
- improvement for larger music recommendation models

By being able to identify nuanced differences in themes, Language of Music hopes to provide a cornerstone to the improvement of AI and ML in the music industry.

## Preprocessing

The preprocessing for Language of Music follows a number of steps to ensure for reliable, noise-minimal topic modelling:

- Normalize casing
- Normalize encoding
- Removal of square brackets and all the content within them.
- Removal of regular brackets and all the content within them.
- Removal of newline escape characters.
- Removal of carriage return characters.
- Removal of adlibs.
- Removal of punctuation.
- Tokenization.
- Part-of-speech tagging.
- Part-of-speech lemmatization.
- Part-of-speech filtering.
- Stopword removal.

The preprocessing makes use of regex and the nltk lib to conduct these steps.