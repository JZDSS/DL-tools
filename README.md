# Installation

## Add Libraries to PYTHONPATH

When running locally, the JZDSS/DL-tools/ directories should be appended to PYTHONPATH. This can be done by running the following from JZDSS/DL-tools/:

```bash
# From jzdss/DL-tools/
export PYTHONPATH=$PYTHONPATH:`pwd`
```

Note: This command needs to run from every new terminal you start. If you wish to avoid running this manually, you can add it as a new line to the end of your ~/.bashrc file (.zshrc if you use zsh rather than bash).