<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![PUll Request][pr-shield]][pr-url]
[![MIT License][license-shield]][license-url]


<br />
<div align="center">
  <a href="https://github.com/Thytu/StockLLM">
    <img src="https://i.ibb.co/rcBKty6/Stock-LLM-min.png" alt="Logo" width="200" height="200">
  </a>

  <h3 align="center">StockLLM</h3>

  <p align="center">
    Elevating Chess Strategy with Fine-Tuned Language Models
    <br />
    <a href="#usage"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="#about-the-project">View Demo</a>
    · <a href="https://github.com/Thytu/StockLLM/issues">Report Bug</a>
    · <a href="https://github.com/Thytu/StockLLM/issues">Request Feature</a>
  </p>
</div>

<br/>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<br/>


## About The Project

StockLLM represents a initiative focusing on refining chess instruction and language modeling through the fine-tuning of a Large Language Model. This project encompasses two pivotal components, each engineered to enhance and streamline the comprehension and dissemination of chess-related knowledge:

### StockLLM (Work in Progress)
StockLLM stands as an ongoing endeavor aimed at developing a highly specialized Large Language Model tailored explicitly for the domain of chess instruction.
StockLLM endeavors to distill and encode intricate chess-related concepts, strategies, and instructional nuances into a language-based model.

Key Features of StockLLM (WIP):

* **Fine-tuned Specialization**: Through meticulous fine-tuning on curated chess instructional datasets, StockLLM seeks to encapsulate the inherent complexities and strategic depth of chess gameplay within its language-based representations.
* **Advanced Contextual Understanding**: StockLLM aims to grasp the subtleties of chess moves, positions, tactics, and strategic principles, fostering an enriched understanding for instructional purposes.
* **Adaptive Learning Capabilities**: The model aspires to adapt dynamically to diverse skill levels, providing tailored guidance, analyses, and instructional content catering to beginners, intermediate, and advanced players alike.

### ChessInstruct Dataset
The [ChessInstruct](https://huggingface.co/datasets/Thytu/ChessInstruct) Dataset serves as the foundation for training and fine-tuning Language Models (LLMs) specifically in the realm of chess instruction.
Derived from the [laion/strategic_game_chess](https://huggingface.co/datasets/laion/strategic_game_chess) dataset, this meticulously curated dataset encompasses a wide array of annotated instructional chess content.

Features of the ChessInstruct Dataset:

* **Rich and Diverse Content**: Curated with a broad spectrum of instructional resources including annotated games, strategic analyses (incoming) and positional evaluations, the dataset facilitates comprehensive learning and modeling.
* **Customizable Training Resource**: The ChessInstruct Dataset allows for the tailored fine-tuning of any Language Model, enabling researchers and practitioners to adapt and optimize LLMs for chess-specific instructional contexts.
* **Annotated Instructional Insights**: Detailed annotations and instructional cues within the dataset provide valuable guidance for language model training, emphasizing strategic moves, tactics, and decision-making processes.

StockLLM, in conjunction with the ChessInstruct Dataset, aims to propel the boundaries of language modeling in the domain of chess instruction.
Through nuanced linguistic representations and tailored instructional datasets, this project envisions revolutionizing the efficacy and depth of chess education by harnessing the power of advanced Natural Language Processing.


<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

TODO

<p align="right">(<a href="#top">back to top</a>)</p>



## Usage

To replicate the experiments, there are two primary methods:

### Using DVC

Utilize DVC to rerun the entire pipeline or modify certain parameters:

Run the entire pipeline:
```bash
dvc repro
```

Play with the parameters used:
```bash
dvc exp run -S "param.to.overwrite=new-value"
```

### Running Individual Stages (without DVC)

Execute a single stage by running the following command:
```bash
python src/entrypoint.py <name-of-stage-to-run>
```

For instance:
```bash
python src/entrypoint.py train-instruct-model
```


Upon executing the DVC pipeline, multiple outputs will be generated and stored in the `outputs/` folder:

1. Intermediate version of [Mistral-7b](https://huggingface.co/mistralai/Mistral-7B-v0.1) : This model incorporates each possible chess move within its vocabulary as a distinct token.
2. [ChessInstruct](https://huggingface.co/datasets/Thytu/ChessInstruct) Dataset: An instructive dataset focused explicitly on chess.
3. StockLLM: An instructive iteration of Mistral-7b specialized specifically for chess.

These outputs encapsulate the refined components developed during the experiment replication process, providing a comprehensive suite of resources for further exploration and utilization within the realm of chess instruction and language modeling.
Feel free to adapt and utilize these outputs in alignment with your specific requirements and research goals.


<p align="right">(<a href="#top">back to top</a>)</p>


## Roadmap

## Roadmap

### Dataset
 - [x] Convert intermediate output format to parquet
 - [x] Generate an intermediate dataset independent of any specific model's prompt format
 - [x] Facilitate local move evaluation by running a StockFish server
 - [ ] Introduce a new task: "detect illegal moves"
 - [ ] Introduce a new task: "calculate WIN/DRAW/LOSE statistics"
 - [ ] Generate strategic game analyses
 - [ ] Implement the use of `LABEL_PROMPT` to enhance the model's output format
 - [ ] Investigate additional datasets to bolster [ChessInstruct](https://huggingface.co/datasets/Thytu/ChessInstruct)
 - [ ] Capture and incorporate strategies employed by various players to enhance the strategic analysis section.
 - [ ] Introduce Time Control Variation: Include datasets or variations that encompass different time controls (blitz, rapid, classical) to diversify the model's exposure to varied game styles.
 - [ ] Conduct an empirical analysis to assess the influence of varying game ELO ratings used during pre-training on the model's performance and adaptability to different skill levels (ELO).

### Model
 - [x] Provide training as a dvc step
 - [ ] Conduct pre-training utilizing a blend of [laion/strategic_game_chess](https://huggingface.co/datasets/laion/strategic_game_chess) and chess-related literature
 - [ ] Perform instruction fine-tuning using a combination of [ChessInstruct](https://huggingface.co/datasets/Thytu/ChessInstruct) and [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
 - [ ] Supply a trained version of StockLLM
 - [ ] Integrate an evaluation step involving matches against Stockfish (across different ELOs)
 - [ ] Determine StockLLM's ELO rating
 - [ ] Experiment with reinforcement learning approaches to allow the model to learn and adapt during gameplay, potentially enhancing its performance.
 - [ ] Focus on improving the model's interpretability by incorporating methods to explain its decisions, especially in strategic analyses or move recommendations.
 
See the [open issues](https://github.com/Thytu/StockLLM/issues) for a full list of proposed features and known issues.


<p align="right">(<a href="#top">back to top</a>)</p>



## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.
If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

Don't forget to give the project a star! 🌟 Thanks again!

Please try to follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

<p align="right">(<a href="#top">back to top</a>)</p>


## Acknowledgments

I extend my heartfelt thanks to [LAION](https://laion.ai/) for graciously providing the [laion/strategic_game_chess](https://huggingface.co/datasets/laion/strategic_game_chess?row=0) dataset that served as the backbone of this project.

<p align="right">(<a href="#top">back to top</a>)</p>



## Contact

Valentin De Matos - [@ThytuVDM](https://twitter.com/ThytuVDM) - vltn.dematos@gmail.com

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/Thytu/StockLLM.svg?style=for-the-badge
[contributors-url]: https://github.com/Thytu/StockLLM/graphs/contributors
[pr-shield]: https://img.shields.io/github/issues-pr/Thytu/StockLLM.svg?style=for-the-badge
[pr-url]: https://github.com/Thytu/StockLLM/pulls
[issues]: https://img.shields.io/github/issues/Thytu/StockLLM
[forks-shield]: https://img.shields.io/github/forks/Thytu/StockLLM.svg?style=for-the-badge&
[forks-url]: https://github.com/Thytu/StockLLM/network/members
[stars-shield]: https://img.shields.io/github/stars/Thytu/StockLLM.svg?style=for-the-badge&
[stars-url]: https://github.com/Thytu/StockLLM/stargazers
[issues-shield]: https://img.shields.io/github/issues/Thytu/StockLLM.svg?style=for-the-badge&
[issues-url]: https://github.com/Thytu/StockLLM/issues
[license-shield]: https://img.shields.io/github/license/Thytu/StockLLM.svg?style=for-the-badge&
[license-url]: https://github.com/Thytu/StockLLM/blob/master/LICENSE
[product-screenshot]: .img/demo-simple.gif
