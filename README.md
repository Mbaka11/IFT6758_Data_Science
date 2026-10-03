# IFT6758 — Data Science

Course materials for IFT6758 (Data Science), part of the Master's in Machine Learning at Mila.

## Project layout

This is the **personal course repository**. The course demos and team project are separate Git clones: they keep their own histories and remotes and are ignored by this repository.

```text
IFT6758_Data_Science/                  # Personal repository
├── README.md
├── IFT6758.code-workspace             # VS Code multi-repository workspace
├── HW/                               # Homework and personal work
├── Slides/                           # Lecture slides
├── data_science/                      # Independent course clone (ignored)
│   ├── demo_1/
│   ├── demo_2/
│   ├── anciens_labs/
│   ├── tutoriels/
│   └── milestone-1/                   # Guidelines in English/French and images
└── workspace/                        # Local repositories (ignored)
    ├── project/                      # Independent team code repository
    └── blog/                         # Planned: separate blog clone, not set up yet
```

| Repository | Local folder | Remote |
|---|---|---|
| Personal course materials | `.` | [Mbaka11/IFT6758_Data_Science](https://github.com/Mbaka11/IFT6758_Data_Science) |
| Course demos and guidelines | `data_science/` | [milarobotlearningcourse/data_science](https://github.com/milarobotlearningcourse/data_science) |
| Team project | `workspace/project/` | [Facu-alfaro/IFT6758-Projet](https://github.com/Facu-alfaro/IFT6758-Projet) |

## Local workspace setup

From this repository's root, clone any **missing** repositories:

```sh
# Skip this command if data_science/ is already cloned.
git clone https://github.com/milarobotlearningcourse/data_science.git data_science

# Skip this command if workspace/project/ is already cloned.
git clone https://github.com/Facu-alfaro/IFT6758-Projet.git workspace/project
```

Open `IFT6758.code-workspace` in VS Code (**File → Open Workspace from File**), or run:

```sh
code IFT6758.code-workspace
```

The workspace opens the personal repository, course clone, and team project together. Run Git commands in the repository you intend to change; a commit or pull in the personal repository does not include or update either child clone. For example:

```sh
git status
git -C data_science status
git -C workspace/project status

# Retrieve published course updates without creating merge commits.
git -C data_science pull --ff-only
```

Do not add the child clones as submodules or force-add their contents to this repository. Commit team code from `workspace/project/`, not from the personal repository. Ignore rules here do **not** prevent commits inside each clone.

Each repository manages its own dependencies and environment. Follow the [course setup instructions](data_science/README_en.md) for demos (`uv sync` from `data_science/`) and the team project's README for project setup; do not assume the personal `.venv` is suitable for both.

## Milestone 1: current scope

See the [English guidelines](data_science/milestone-1/README_en.md) or [French guidelines](data_science/milestone-1/README_fr.md).

- **Now:** workspace setup, then **Data Acquisition (25%)**: regular-season and playoff play-by-play data for **2016–17 through 2023–24**, plus only the corresponding acquisition tutorial in the blog.
- **Later:** the IFT6758-required **LLM/RAG** acquisition work and manual-versus-LLM comparison. These are deferred, not skipped.
- **Blog:** set up a separate repository under `workspace/blog/` later, based on the [Jekyll blog template](https://github.com/milarobotlearningcourse/data_science_blog). Do not copy it into the code repository; the milestone requires separate blog and codebase ZIPs.
- **Project template:** the [official code template](https://github.com/milarobotlearningcourse/data_science_project) is a reference; work in the team's existing clone rather than creating another project repository.

Do not commit downloaded NHL data or large generated files. Add data/cache ignore rules **inside the team repository** when setting up acquisition. The guidelines require the manual comparison baseline to be written by the team without LLM-generated code and require disclosure of AI assistance.
