# AI Hedge Fund

This is a proof of concept for an AI-powered hedge fund.  The goal of this project is to explore the use of AI to make trading decisions.  This project is for **educational** purposes only and is not intended for real trading or investment.

This system employs several agents working together:

1. Ben Graham Agent - The godfather of value investing, only buys hidden gems with a margin of safety
2. Bill Ackman Agent - An activist investors, takes bold positions and pushes for change
3. Catherine Wood Agent - The queen of growth investing, believes in the power of innovation and disruption
4. Warren Buffett Agent - The oracle of Omaha, seeks wonderful companies at a fair price
5. Valuation Agent - Calculates the intrinsic value of a stock and generates trading signals
6. Sentiment Agent - Analyzes market sentiment and generates trading signals
7. Fundamentals Agent - Analyzes fundamental data and generates trading signals
8. Technicals Agent - Analyzes technical indicators and generates trading signals
9. Risk Manager - Calculates risk metrics and sets position limits
10. Portfolio Manager - Makes final trading decisions and generates orders

<img width="1117" alt="Screenshot 2025-02-09 at 11 26 14 AM" src="https://github.com/user-attachments/assets/16509cc2-4b64-4c67-8de6-00d224893d58" />


**Note**: the system simulates trading decisions, it does not actually trade.

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No warranties or guarantees provided
- Past performance does not indicate future results
- Creator assumes no liability for financial losses
- Consult a financial advisor for investment decisions

By using this software, you agree to use it solely for learning purposes.

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
  - [Running the Hedge Fund](#running-the-hedge-fund)
  - [Running the Backtester](#running-the-backtester)
- [Data Sources](#data-sources)
- [Model Providers](#model-providers)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Feature Requests](#feature-requests)
- [License](#license)

## Setup

Clone the repository:
```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Set up your environment variables:
```bash
# Create .env file for your API keys
cp .env.example .env
```

4. Set your API keys:
```bash
# For running LLMs hosted by openai (gpt-4o-mini, gpt-4o-mini-mini, etc.)
# Get your OpenAI API key from https://platform.openai.com/
OPENAI_API_KEY=your-openai-api-key

# For running LLMs hosted by groq (deepseek, llama3, etc.)
# Get your Groq API key from https://groq.com/
GROQ_API_KEY=your-groq-api-key

# For getting financial data to power the hedge fund
# Get your Financial Datasets API key from https://financialdatasets.ai/
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key

# For using Ollama models (optional, defaults to http://localhost:11434)
# OLLAMA_BASE_URL=http://localhost:11434
```

**Important**: You must set `OPENAI_API_KEY`, `GROQ_API_KEY`, or `ANTHROPIC_API_KEY` for the hedge fund to work if you want to use cloud-based LLMs. If you want to use local models through Ollama, you need to have Ollama running on your machine.

Financial data for AAPL, GOOGL, MSFT, NVDA, and TSLA is free and does not require an API key.

For any other ticker, you will need to set the `FINANCIAL_DATASETS_API_KEY` in the .env file or use Yahoo Finance as the data source.

## Usage

### Running the Hedge Fund
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

**Example Output:**
<img width="992" alt="Screenshot 2025-01-06 at 5 50 17 PM" src="https://github.com/user-attachments/assets/e8ca04bf-9989-4a7d-a8b4-34e04666663b" />

You can also specify a `--show-reasoning` flag to print the reasoning of each agent to the console.

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --show-reasoning
```
You can optionally specify the start and end dates to make decisions for a specific time period.

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 
```

To use Yahoo Finance as the data source instead of the API:

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --data-source yahoo_finance
```

### Running the Backtester

```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

**Example Output:**
<img width="941" alt="Screenshot 2025-01-06 at 5 47 52 PM" src="https://github.com/user-attachments/assets/00e794ea-8628-44e6-9a84-8f8a31ad3b47" />

You can optionally specify the start and end dates to backtest over a specific time period.

```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01
```

To use Yahoo Finance as the data source for backtesting:

```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --data-source yahoo_finance
```

## Data Sources

The system supports two data sources:

1. **Financial Datasets API** (default): A comprehensive financial data API. Requires an API key set in your `.env` file as `FINANCIAL_DATASETS_API_KEY`.

2. **Yahoo Finance**: An alternative data source that doesn't require an API key. Use with the `--data-source yahoo_finance` flag.

## Model Providers

The system supports multiple AI model providers:

1. **OpenAI**: Cloud-based models like GPT-4o mini. Requires `OPENAI_API_KEY`.

2. **Anthropic**: Claude models like Claude 3.5 Sonnet. Requires `ANTHROPIC_API_KEY`.

3. **Groq**: Models like llama-3.3 70b. Requires `GROQ_API_KEY`.

4. **Ollama**: Local models that run on your machine. Requires Ollama to be installed and running. Supports models like:
   - llama3 (various sizes)
   - mistral
   - gemma
   - phi3

To use Ollama models:
1. Install Ollama from [https://ollama.com/](https://ollama.com/)
2. Pull the models you want to use: `ollama pull llama3`
3. Make sure Ollama is running in the background
4. Select an Ollama model when prompted by the hedge fund system

## Project Structure 
```
ai-hedge-fund/
├── src/
│   ├── agents/                   # Agent definitions and workflow
│   │   ├── bill_ackman.py        # Bill Ackman agent
│   │   ├── fundamentals.py       # Fundamental analysis agent
│   │   ├── portfolio_manager.py  # Portfolio management agent
│   │   ├── risk_manager.py       # Risk management agent
│   │   ├── sentiment.py          # Sentiment analysis agent
│   │   ├── technicals.py         # Technical analysis agent
│   │   ├── valuation.py          # Valuation analysis agent
│   │   ├── warren_buffett.py     # Warren Buffett agent
│   ├── tools/                    # Agent tools
│   │   ├── api.py                # Financial API tools
│   │   ├── yahoo_finance.py      # Yahoo Finance data client
│   │   ├── data_source.py        # Data source manager
│   ├── backtester.py             # Backtesting tools
│   ├── main.py # Main entry point
├── pyproject.toml
├── ...
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Important**: Please keep your pull requests small and focused.  This will make it easier to review and merge.

## Feature Requests

If you have a feature request, please open an [issue](https://github.com/virattt/ai-hedge-fund/issues) and make sure it is tagged with `enhancement`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
