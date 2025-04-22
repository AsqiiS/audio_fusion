# Audio_fusion
Implementation of the transfusion (https://arxiv.org/html/2408.11039v1) based approach, but applied on audios and flow-matching is used instead of regular diffusion. 


## Training setup: 
The model is trained using train.py, where dataset and model parameters can be configured. 

## Evaluation: 
Text generation is evaluated on evaluate_text.py and audio is evaluated on evaluate_sim.py 

## Contributions 
This code is adapted from multiple papers and implementations, like: https://github.com/SWivid/F5-TTS, https://github.com/lucidrains/transfusion-pytorch, https://github.com/MonoFormer/MonoFormer/tree/main 
