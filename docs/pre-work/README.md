# Lab 0: Workshop Pre-Work

Welcome to the Output Drift workshop! This lab will guide you through setting up your environment and ensuring you have all the prerequisites installed.

## Prerequisites

Before you begin, make sure you have the following installed on your system:

### Required Software

1. **Python 3.11 or higher**

   Check your Python version:
   ```bash
   python --version
   # or
   python3 --version
   ```

   If you need to install Python, visit [python.org](https://www.python.org/downloads/)

2. **Git**

   Check if Git is installed:
   ```bash
   git --version
   ```

   If you need to install Git, visit [git-scm.com](https://git-scm.com/downloads)

3. **Text Editor or IDE**

   We recommend:
   - VS Code
   - PyCharm
   - Jupyter Notebook/Lab
   - Or any editor of your choice

### Optional (but Recommended)

1. **Ollama** (for free, local LLM testing)

   Install from [ollama.ai](https://ollama.ai/)

   After installation, verify:
   ```bash
   ollama --version
   ```

   Pull a recommended model:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ```

## Step 1: Clone the Repository

Clone the workshop repository to your local machine:

```bash
git clone https://github.com/ibm-client-engineering/output-drift-financial-llms
cd output-drift-financial-llms
```

For reproducibility, checkout the v0.1.0 release:

```bash
git checkout v0.1.0
```

## Step 2: Set Up a Virtual Environment

Create and activate a Python virtual environment:

=== "macOS/Linux"
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

=== "Windows"
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

You should see `(venv)` in your terminal prompt, indicating the virtual environment is active.

## Step 3: Install Dependencies

Install all required Python packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- `openai` - OpenAI API client
- `anthropic` - Anthropic API client
- `ibm-watsonx-ai` - IBM watsonx.ai client
- `pandas` - Data analysis
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization
- `python-dotenv` - Environment variable management
- And other dependencies

## Step 4: Verify Installation

Test that the framework can import correctly:

```bash
python -c "from harness.runner import DriftRunner; print('✅ Installation successful!')"
```

If you see "✅ Installation successful!", you're all set!

## Step 5: Set Up API Keys (Optional)

To run experiments with cloud providers, you'll need API keys. **Don't worry if you don't have all of these**—you can start with Ollama (free and local) and add others later.

### Create a `.env` File

In the repository root, create a `.env` file:

```bash
touch .env
```

### Add Your API Keys

Edit the `.env` file and add the keys you have:

```bash
# Ollama (local, no key needed)
OLLAMA_BASE_URL=http://localhost:11434

# IBM watsonx.ai (if you have access)
WATSONX_API_KEY=your_watsonx_api_key_here
WATSONX_PROJECT_ID=your_project_id_here

# OpenAI (if you have access)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic (if you have access)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

!!! warning "Keep Your Keys Secret"
    Never commit your `.env` file to Git! It's already in `.gitignore` to prevent accidental commits.

### Where to Get API Keys

=== "Ollama (Free)"
    **No API key needed!** Just install and run locally.

    ```bash
    # Install from https://ollama.ai/
    ollama serve
    ```

=== "IBM watsonx.ai"
    1. Visit [watsonx.ai](https://www.ibm.com/watsonx)
    2. Sign up for a trial or use your IBM Cloud account
    3. Create a project and get your API key and project ID
    4. [Setup Guide](https://www.ibm.com/docs/en/watsonx)

=== "OpenAI"
    1. Visit [platform.openai.com](https://platform.openai.com/)
    2. Sign up and navigate to API keys
    3. Create a new API key
    4. Add billing information (required for API access)

=== "Anthropic"
    1. Visit [console.anthropic.com](https://console.anthropic.com/)
    2. Sign up and navigate to API keys
    3. Create a new API key
    4. Add billing information

## Step 6: Test Your Setup

Run the environment test script to verify everything is configured correctly:

```bash
python examples/test_setup.py
```

Expected output:

```
🔍 Testing Output Drift Framework Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Python version: 3.11.0
✅ Dependencies installed
✅ Framework imports working
✅ Environment variables loaded

Provider Availability:
✅ Ollama: Available (http://localhost:11434)
⚠️  watsonx.ai: API key not configured
⚠️  OpenAI: API key not configured
⚠️  Anthropic: API key not configured

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Setup complete! You're ready for Lab 1.
```

!!! success "At Least One Provider"
    You need at least one provider configured (even just Ollama) to complete the workshop!

## Troubleshooting

### Python Version Issues

If you have multiple Python versions installed:

```bash
# Use python3.11 explicitly
python3.11 -m venv venv
```

### Ollama Connection Issues

If Ollama isn't responding:

```bash
# Start Ollama server
ollama serve

# In another terminal, test:
curl http://localhost:11434/api/tags
```

### Import Errors

If you see import errors, ensure you're in the virtual environment:

```bash
# Check if venv is activated
which python
# Should show: /path/to/your/repo/venv/bin/python

# If not, activate it:
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### Permission Errors on macOS/Linux

If you encounter permission errors:

```bash
# Use --user flag
pip install --user -r requirements.txt
```

## Next Steps

Once your environment is set up and tested:

1. **Proceed to [Lab 1: Understanding Output Drift](../lab-1/README.md)**
2. Explore the `examples/` directory for sample configurations
3. Review the research paper in `docs/resources/paper.md`

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](../resources/troubleshooting.md)
2. Ask workshop facilitators
3. Open an [issue on GitHub](https://github.com/ibm-client-engineering/output-drift-financial-llms/issues/new)

---

!!! success "Ready?"
    If all tests pass, you're ready to move on to [Lab 1: Understanding Output Drift](../lab-1/README.md)!
