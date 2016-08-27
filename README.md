# statebrain

This is a probabilistic model that learns to generate sequences. I had this idea while thinking about what an LSTM *really* has over a Markov chain: an LSTM might not store much more information than a Markov chain, but it can choose what information to store (e.g. long-term dependencies). Thus, I had the idea to make a Markov chain where the states' meanings can be learned, so that they could theoretically represent long-term information.

I have not studied Hidden Markov Models, but I have read the first paragraph to their Wikipedia page before (probably out of interest in speech recognition). A few hours after having the idea for statebrain, it occurred to me that statebrain is really just a way to derive latent variables. As a result, this may be equivalent to HMMs.
