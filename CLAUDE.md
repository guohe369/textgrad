# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TextGrad is an automatic differentiation framework for text optimization using large language models. It implements "textual gradients" through LLM feedback, following a PyTorch-like API for optimizing prompts, solutions, code, and other text-based variables.

## Essential Commands

### Installation and Setup
```bash
# Install from PyPI
pip install textgrad

# Install with vLLM support
pip install textgrad[vllm]

# Install bleeding edge from source
pip install git+https://github.com/zou-group/textgrad.git

# Install development requirements
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_basics.py
python -m pytest tests/test_engines.py
python -m pytest tests/test_api.py

# Run with verbose output
python -m pytest -v tests/
```

### Development Setup
```bash
# Install in development mode
pip install -e .

# Set up environment variables for API keys
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
```

## Core Architecture

### Engine System
The engine system (`textgrad/engine/`) supports multiple LLM providers:
- **OpenAI**: GPT-4, GPT-3.5-turbo with multimodal support
- **Anthropic**: Claude models (Opus, Sonnet, Haiku) with multimodal support
- **Google**: Gemini models
- **Local**: vLLM, Ollama, Together AI
- **Experimental**: LiteLLM for broader provider support

Engine initialization uses the `get_engine(engine_name, **kwargs)` function with shortcuts:
- `"opus"` → `"claude-3-opus-20240229"`
- `"sonnet"` → `"claude-3-sonnet-20240229"`
- `"sonnet-3.5"` → `"claude-3-5-sonnet-20240620"`

### Core Components

#### Variables (`textgrad/variable.py`)
- `Variable`: Core data structure for text that can be optimized
- Supports gradient computation with `requires_grad=True`
- Role descriptions guide optimization behavior

#### Autograd System (`textgrad/autograd/`)
- `llm_ops.py`: LLM-based operations for forward/backward passes
- `function.py`: Function abstraction for differentiable operations
- `algebra.py`: Mathematical operations on text variables
- Backward pass prompts in `llm_backward_prompts.py`

#### Optimizers (`textgrad/optimizer/`)
- `TGD` (Textual Gradient Descent): Main optimizer
- `GuidanceOptimizer`: Structured optimization with guidance
- Follows PyTorch optimizer API pattern

#### Loss Functions (`textgrad/loss.py`)
- `TextLoss`: Natural language loss function specification
- Custom loss functions can be defined for specific evaluation criteria

#### Models (`textgrad/model.py`)
- `BlackboxLLM`: Wrapper for LLM engines
- Supports system prompts and parameter optimization

### Configuration System

#### Engine Configuration (`textgrad/config.py`)
- `set_backward_engine(engine, override=False)`: Set global backward engine
- `SingletonBackwardEngine`: Manages global backward engine state
- Supports engine override for different optimization steps

#### Logging (`textgrad/config.py`)
- JSON-formatted logging to `./logs/` directory
- Controlled by `TEXTGRAD_LOG_DIR` environment variable
- Automatic log file rotation by timestamp

## Evaluation Framework

### Built-in Tasks (`textgrad/tasks/`)
- `gsm8k.py`: Mathematical reasoning evaluation
- `mmlu.py`: Multi-task language understanding
- `big_bench_hard.py`: Challenging reasoning tasks
- `gpqa.py`: Graduate-level question answering
- `leetcode.py`: Code optimization tasks

### Evaluation Scripts (`evaluation/`)
- `prompt_optimization.py`: Automated prompt optimization
- `solution_optimization.py`: Solution quality improvement
- `code_optimization/`: LeetCode code optimization with test-time supervision

### Multimodal Support (`textgrad/tasks/multimodal/`)
- `mathvista.py`: Mathematical visual reasoning
- `scienceqa.py`: Science question answering with images
- Image utilities in `textgrad/utils/image_utils.py`

## API Usage Patterns

### Basic Optimization
```python
import textgrad as tg

# Set backward engine
tg.set_backward_engine("gpt-4o", override=True)

# Create optimizable variable
variable = tg.Variable("text to optimize", requires_grad=True)

# Define loss and optimizer
loss_fn = tg.TextLoss("evaluation criteria")
optimizer = tg.TGD(parameters=[variable])

# Optimization loop
loss = loss_fn(variable)
loss.backward()
optimizer.step()
```

### Engine Management
```python
# Standard engines
engine = tg.get_engine("gpt-4o")
engine = tg.get_engine("claude-3-opus-20240229")

# Experimental engines (broader provider support)
engine = tg.get_engine("experimental:gpt-4o", cache=False)

# Local engines
engine = tg.get_engine("vllm-meta-llama/Meta-Llama-3-8B-Instruct")
```

## Testing Strategy

Tests are organized in `tests/`:
- `test_basics.py`: Core functionality with dummy engines
- `test_engines.py`: Engine-specific functionality
- `test_api.py`: High-level API testing

Use `DummyEngine` and `IdempotentEngine` classes for testing without API calls.

## Performance Considerations

- Caching is available for experimental engines: `cache=True`
- Multimodal engines automatically detected for image processing
- Logging can be disabled: `logging.disable(logging.CRITICAL)`
- Use `diskcache` for persistent caching across sessions

## Memoranda

- 始终使用llm-prealign这个环境