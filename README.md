# QuickEmbedding
An improvement on Textual Inversion embedding training leveraging a clip loss. 
As noted in CustomDiffusion, Textual inversion suffers from poor image/text alignment compared to other methods while also taking considerably long to train with a loss that does not have such a smooth convergence.
For that reason I propose splitting the training run into two phases, where the first phase is just a cosine similarity loss between CLIP text and image outputs. I precompute the embeddings, so the only model in memory is the text encoder.
Thus, This can be done extremely quickly and requires little VRAM.

In fact, this actually works pretty well on its own, but it doesn't entirely cater to how the diffusion model ultimately handles things, so we can do a second phase of training which is just vanilla textual inversion.
We can think of the first phase as getting us 99% of the way there and letting regular textual inversion do the last 1% for us.
