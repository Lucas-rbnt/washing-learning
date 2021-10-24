# Contributing to Washing Learning

Everything you need to know to contribute efficiently to the project.

## Project structure and conventions

### Codebase structure

Everything is contained into Washing Learning and is then divided by field of applications, vision, language, reinforcement learning,
time series...


### Style conventions

-   **Code**:
    -   Use type hints for every functions ([type hints cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html))
    -   Format your code using the [black](https://github.com/psf/black) auto-formatter
    -   Ensure to document your code using type hints compatible docstrings. In doing so, please follow [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) so it can ease the process of documentation later.

-   **Commit message**: please follow [Udacity guide](http://udacity.github.io/git-styleguide/)

## Contributing to the project

In order to contribute to project, you will first need to **set up your development environment** and then follow the **contributing workflow** and the **code & commit guidelines**.

-   [Project Setup](#project-setup): *fork the project and install dependencies in your own development environment*

    -  [**Create a virtual environment**](#create-a-virtual-environment)
    -  [**Fork the mlutils project**](#fork-the-repository)
    -  [**Set origin and upstream remotes**](#set-origin-and-upstream-remotes-repositories)
    -  [**Install project dependencies**](#install-project-dependencies)




### Project Setup

* * *

#### 1. Create a virtual environment

<br>

-   We are going to create a python3.6 virtual environment dedicated to the `mlutils` project using [conda](https://docs.conda.io/en/latest/) as an environment management system.

    ```shell
    conda create --name washing-learning python=3.6 anaconda
    conda activate washing-learning
    ```

#### 2. Fork the repository

<br>

-   In order to obtain your own copy of the project.

    -  Create a fork by clicking on the **fork button** on the current repository page

    -  Clone your fork locally.

        ```shell
        cd /PATH_WASHING_LEARNING
        git clone https://github.com/YOUR_USERNAME/washing-learning.git
        cd washing-learning
        ```

#### 3. Set origin and upstream remotes repositories

<br>

1.  Configure your fork `YOUR_USERNAME/washing-learning` as `origin` remote

2.  Configure `washing-learning repository` as `upstream` remote

    ```shell
    git remote add upstream URL
    git pull --rebase upstram master
    ```

#### 4. Install project dependencies

<br>

```shell
pip install .
#or in editable mode
pip install -e .
```
