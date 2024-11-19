# git

git - the stupid content tracker

## Glossary

- **index** — see _staging area_
- **staging area** — where you prepare changes before committing
- **working tree** — directory that contains project's files

## Apply a patch

> **$ man 1 git-clean**
> git-apply - Apply a patch to files and/or to the index

```bash
git apply
```

**Flags**
- `--stat`: only output diffstat
- `--numstat`: like `--stat` but easier to parse for scripts
- `--summary`: only output information from <abbr title="Stuff like `diff --git a/file1 b/file1`, `@@ -l,s +l,s @@`, etc">git diff extension headers</abbr>
- `--check`: only see if the patch is applicable
- `--reject`: apply as much as possible, creates .rej files


## Remove untracked files

> **$ man 1 git-clean**
> git-clean - Remove untracked files from the working tree

```bash
git clean
```

**Flags**
- `-n` dry run
- `-f` remove untracked files
- `-fd` remove untracked directories
- `-fdX` remove ignored files
- `-fx` remove untracked and ignored files