# Create mamba env
micromamba create -n ai-agent python=3.11
micromamba activate ai-agent

# install library
pip install -r requirements.txt

# Directory structure
ai-agent/
├── app.py
├── requirements.txt
├── .env
├── tools/
│   ├── calculator.py
│   └── weather.py
└── tests/


# Description
LCEL is primarily for composing chains/runnables; an agent is a control loop around an LLM and tools.

