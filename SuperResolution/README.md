## Scaling up GANs for Text-to-Image Synthesis (things to go in discussions)

### Details from the paper -- 
supports latent space editing applications such as - latent interpolation, style mixing, and vector arithmetic operations

Diffusion Process --
This is a double-edged sword, as iterative methods enable stable training with simple objectives but incur a high computational cost during inference

simply scaling the backbone causes unstable training

we effectively scale the generator’s capacity by re-
taining a bank of filters and taking a sample-specific linear
combination.

interleaving both
self-attention (image-only) and cross-attention (image-text)
with the convolutional layers improves performance

We first generate at 64 × 64 and then upsample to
512 × 512. 

64*4 = 256

### Architecture --
G(z,c) -> x E H*W*3(4?) z ~ normal dist, text conditioning singal C
D(x,c) 


CLIP(F) -> text encoder -> local features -> cross attention(conv)


G = G.M
w = M(z,c) -> maps to style vector w, which in turns modulated a series of upsampling conv layers in the synthesis network G(w)

w is the only source of information to the network 

disentangled and continuous latent space - A continuous latent space means that small changes in the latent variables (the input to the generator) lead to small and smooth changes in the generated output

For example, in a disentangled latent space for faces, one dimension might control the hair color, another the lighting, another the pose, and so on


### Training Details -- 
