# Probabilistic language guided attention for Classification models 


  
  
## About the Project
This is a project by Svante Andersson, Olle Flygar, Leo Kemetli and Edvard Schmidt. It was supervised by Prashant Singh and Li Ju at Uppsala University. The project investigates the integration of
probabilistic embeddings in a language guided attention learning task.

<!-- Table of Contents -->
## Table of Contents
  - [Introduction](#introduction)
    
  - [Paper](#paper)

  - [Prerequisites](#prerequisites)

  - [Contact](#contact)
  - [References](#references)
  - [License](#license)

---
## Introduction
This is the PyTorch implementation the project. The point of entry in this repository is the ProbVLM folder for training the adapter to later be used for attention map generation. Run the
`train_ProbVLM_...` appropariate for the model and dataset. Then attention maps can be generated throught the `Attention_maps_for...` file.

In the BayesVLM framework one can directly run the attention map generating file 'attention_waterXX_Bayes.py' to generate attention maps via the RISE based approach using a 500-mask based saliency map.

Finnally, to run the downstream task, run `main.py` in the GALS folder with adequate parsers. See references for a thorough walkthrough.



## Paper
TBA

---

<!-- Prerequisites -->
## Prerequisites
To run the ProbVLM files, run 
```bash 
  conda install -r requirements.txt
```
Use pip to install the following packages also necesary: XX, XX, XX, XX.
The BayesVLM code can be run on the same requirements.
Use the requirements.txt found in GASL in a similar manor along with the config files (.yaml) found in '/configs' foun in the GALS folder.
 

---
<!-- Contact -->
## Contact
Svante Andersson - [@linkedin](https://www.linkedin.com/in/svante-andersson-673b2921a/)  
Olle Flygar - [@linkedin](https://www.linkedin.com/in/olle-flygar-2769a3325/)  
Leo Kemetli - [@linkedin](https://www.linkedin.com/in/leo-lindström-kemetli-552a30290/)  
Edvard Schmidt - [@linkedin](https://www.linkedin.com/in/edvard-schmidt-05a014326/)  

<!-- Links -->
## References

The project was inspired and built upon these projects:

[1] Nautiyal, M., Gheorghe, S. A., Stefa, K., Ju, L., Sintorn, I.-M., & Singh, P. (2025). *PARIC: Probabilistic Attention Regularization for Language-Guided Image Classification from Pre-trained Vision Language Models*. arXiv preprint arXiv:2503.11360.

[2] Baumann, A., Li, R., Klasson, M., Mentu, S., Karthik, S., Akata, Z., Solin, A., & Trapp, M. (2024). *Post-hoc Probabilistic Vision-Language Models*. arXiv preprint arXiv:2412.06014.

[3] Petryk, S., Dunlap, L., Nasseri, K., Gonzalez, J., Darrell, T., & Rohrbach, A. (2022). *On Guiding Visual Attention with Language Specification*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

If used, please cite the following:

```bibtex
@article{nautiyal2025paric,
  title={PARIC: Probabilistic Attention Regularization for Language-Guided Image Classification from Pre-trained Vision Language Models},
  author={Nautiyal, Mayank and Gheorghe, Stela Arranz and Stefa, Kristiana and Ju, Li and Sintorn, Ida-Maria and Singh, Prashant},
  journal={arXiv preprint arXiv:2503.11360},
  year={2025}
}

@article{baumann2024bayesvlm,
  title={Post-hoc Probabilistic Vision-Language Models},
  author={Baumann, Anton and Li, Rui and Klasson, Marcus and Mentu, Santeri and Karthik, Shyamgopal and Akata, Zeynep and Solin, Arno and Trapp, Martin},
  journal={arXiv preprint arXiv:2412.06014},
  year={2024}
}

@inproceedings{petryk2022gals,
  title={On Guiding Visual Attention with Language Specification},
  author={Petryk, Suzanne and Dunlap, Lisa and Nasseri, Keyan and Gonzalez, Joseph and Darrell, Trevor and Rohrbach, Anna},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```







<!-- License -->
## License

```
MIT License

Copyright (c) 2025 Svante Andersson, Olle Flygar, Leo Kemetli, Edvard Schmidt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
