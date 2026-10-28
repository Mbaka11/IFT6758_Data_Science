# IFT6758 — Data Science

Course materials for IFT6758 (Data Science), part of the Master's in Machine Learning at Mila.

## Project layout

This is the **personal course repository**. The course demos, team project, and team blog are separate Git clones: they keep their own histories and remotes and are ignored by this repository.

```text
IFT6758_Data_Science/                  # Personal repository
├── README.md
├── IFT6758.code-workspace             # VS Code multi-repository workspace
├── HW/                               # Homework and personal work
├── Slides/                           # Lecture slides
├── notes/                            # Local study/context notes (ignored)
├── data_science/                      # Independent course clone (ignored)
│   ├── demo_1/
│   ├── demo_2/
│   ├── anciens_labs/
│   ├── tutoriels/
│   └── milestone-1/                   # Guidelines in English/French and images
└── workspace/                        # Local repositories (ignored)
    ├── project/                      # Independent team code repository
    └── blog/                         # Independent private team Jekyll blog
```

| Repository | Local folder | Remote |
|---|---|---|
| Personal course materials | `.` | [Mbaka11/IFT6758_Data_Science](https://github.com/Mbaka11/IFT6758_Data_Science) |
| Course demos and guidelines | `data_science/` | [milarobotlearningcourse/data_science](https://github.com/milarobotlearningcourse/data_science) |
| Team project | `workspace/project/` | [Facu-alfaro/IFT6758-Projet](https://github.com/Facu-alfaro/IFT6758-Projet) |
| Team blog (private) | `workspace/blog/` | [Mbaka11/IFT6758-Blog](https://github.com/Mbaka11/IFT6758-Blog) |

## Local workspace setup

From this repository's root, clone any **missing** repositories:

```sh
# Skip this command if data_science/ is already cloned.
git clone https://github.com/milarobotlearningcourse/data_science.git data_science

# Skip this command if workspace/project/ is already cloned.
git clone https://github.com/Facu-alfaro/IFT6758-Projet.git workspace/project

# Skip this command if workspace/blog/ is already cloned. Requires team access.
git clone https://github.com/Mbaka11/IFT6758-Blog.git workspace/blog
```

Open `IFT6758.code-workspace` in VS Code (**File → Open Workspace from File**), or run:

```sh
code IFT6758.code-workspace
```

The workspace opens the personal repository, course clone, team project, and team blog together. Run Git commands in the repository you intend to change; a commit or pull in the personal repository does not include or update any child clone. For example:

```sh
git status
git -C data_science status
git -C workspace/project status
git -C workspace/blog status

# Retrieve published course updates without creating merge commits.
git -C data_science pull --ff-only
```

Do not add the child clones as submodules or force-add their contents to this repository. Commit team code from `workspace/project/` and blog changes from `workspace/blog/`, not from the personal repository. Ignore rules here do **not** prevent commits inside each clone.

Each repository manages its own dependencies and environment. Follow the [course setup instructions](data_science/README_en.md) for demos (`uv sync` from `data_science/`) and the team project's README for project setup; do not assume the personal `.venv` is suitable for both. The blog uses Ruby/Bundler, not the Python environment.

## Team project Python environment

On this machine, **uv 0.12.22** created `workspace/project/.venv/` with **Python 3.11.17**, the editable `ift6758` package, and the project's locked dependencies/development tools. Core imports and JupyterLab were verified. Qwen/LLM extras are not installed.

- `pyproject.toml` declares the project requirements.
- `uv.lock` records exact resolved dependency versions; it was not changed during setup.
- `.venv/` is the local isolated environment and is ignored by Git. It is separate from the personal and course/demo environments.
- Use `uv run` from `workspace/project/`; manual environment activation is not necessary.

To reproduce the setup on another machine with uv installed, run `uv sync --locked --python 3.11` from the team project root. On this machine, open a new PowerShell terminal and launch Jupyter with:

```powershell
# Refresh PATH if this terminal was open before uv was installed.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + $env:Path
Set-Location "D:\School\IFT6758_Data_Science\workspace\project"
uv run --locked jupyter lab
```

For VS Code notebooks, select the interpreter/kernel at `workspace/project/.venv/Scripts/python.exe`, not the personal repository's environment.

## Team blog and local preview

The private [team blog repository](https://github.com/Mbaka11/IFT6758-Blog) was initialized from the [official template](https://github.com/milarobotlearningcourse/data_science_blog), preserving its Git history and MIT license. `origin` points to the private team repository; `upstream` in this local clone points to the public template. Push your team's changes to `origin`, not `upstream`.

Write milestone reports in `workspace/blog/_posts/`. Keep Python analysis in `workspace/project/` and copy only the images/exported HTML figures needed for the report into the blog. The two repositories do not synchronize automatically.

On this Windows machine, **Ruby+Devkit 3.3.12**, **Bundler 4.0.22**, and **Jekyll 4.4.1** are installed. Ruby is at `C:/Ruby33-x64`; the template passed `bundle exec jekyll build`. Open a new terminal to pick up Ruby's updated PATH (restart VS Code if `ruby` is still not found).

Other teammates first need **Ruby 3.2+** (Ruby+Devkit on Windows), following the [blog installation guide](workspace/blog/README.md#installation). Then run this one-time dependency setup from this repository's root:

```sh
cd workspace/blog
gem install bundler
bundle config set --local path vendor/bundle
bundle install
```

Gems live in the blog's ignored `vendor/bundle/` directory. `.bundle/`, `Gemfile.lock`, `.jekyll-cache/`, and generated `_site/` files are also ignored according to the template's policy.

To preview on this machine, open a new terminal at this repository's root and run:

```sh
cd workspace/blog
bundle exec jekyll serve
```

Open `http://127.0.0.1:4000/`; stop the server with `Ctrl+C`. Keep GitHub Pages disabled: the repository and report stay private, and the site is previewed locally.

## Team feature-branch workflow

Both team repositories are checked out on **`feature/milestone-1-data-acquisition`**, which is also pushed to their respective GitHub remotes. The branches share a name, not a Git history. Do not commit or push directly to `main`.

Check the current branches before working:

```sh
git -C workspace/project branch --show-current
git -C workspace/blog branch --show-current
```

Commit small, understood changes in the relevant repository, then back up those commits to its feature branch:

```sh
git -C workspace/project push origin feature/milestone-1-data-acquisition
git -C workspace/blog push origin feature/milestone-1-data-acquisition
```

Pull requests are deferred until there are meaningful code/report changes to review. Merge reviewed work into `main` before the final milestone ZIP submissions, as required by the course.

## Milestone 1: current scope

See the [English guidelines](data_science/milestone-1/README_en.md) or [French guidelines](data_science/milestone-1/README_fr.md).

- **Now:** the workspace, blog dependencies, and project Python environment are ready. Begin **Data Acquisition (25%)**: regular-season and playoff play-by-play data for **2016–17 through 2023–24**, plus only the corresponding acquisition tutorial in the blog. Acquisition code has not been written.
- **Later:** the IFT6758-required **LLM/RAG** acquisition work and manual-versus-LLM comparison. These are deferred, not skipped.
- **Blog:** the private team repository is cloned at `workspace/blog/` and its template builds locally. Report content has not been written. Keeping blog and code in sibling repositories makes the milestone's separate blog and codebase ZIP submissions straightforward.
- **Project template:** the [official code template](https://github.com/milarobotlearningcourse/data_science_project) is a reference; work in the team's existing clone rather than creating another project repository.

Do not commit downloaded NHL data or large generated files. Add data/cache ignore rules **inside the team repository** when setting up acquisition. The guidelines require the manual comparison baseline to be written by the team without LLM-generated code and require disclosure of AI assistance.

Detailed personal explanations and the next learning checkpoint are kept in the local, Git-ignored `notes/` folder. These notes are not included in GitHub clones.
