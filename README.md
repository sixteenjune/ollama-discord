# Discord Ollama Bot

Locally-Hosted discord bot to connect to Ollama

## Features

- **Local AI Generation:** Connects directly to a local Ollama `api/chat` endpoint. No external API keys required!
- **Persistent Memory:** Automatically extracts conversational facts (like names and user preferences) and saves them to a local `.sqlite` database.
- **Contextual History:** Keeps a rolling window of recent messages for natural, conversational flow.
- **Rate Limiting:** Built-in safeguards to prevent API spam.
- **Docker / CasaOS Ready:** Includes a `docker-compose.example.yml` configured for easy deployment on CasaOS or standard Docker environments.

## Prerequisites

- [Python 3.11+](https://www.python.org/downloads/) (if running locally)
- [Ollama](https://ollama.com/) installed and running.
- Setup a model by editing `OLLAMA_MODEL` in `main.py`).
- A [Discord Bot Token](https://discord.com/developers/applications).

## Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sixteenjune/ollama-discord.git
   cd ollama-discord
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your environment variables:**
   You can export these in your terminal or parse them however your setup prefers (note: `.env` files are ignored by git).
   ```bash
   export DISCORD_TOKEN="your_discord_bot_token"
   
   # Optional: These will default to the values below if not provided
   export OLLAMA_URL="http://localhost:11434/api/chat"
   export MEMORY_FILE="user_memory.sqlite"
   ```

4. **Run the bot:**
   ```bash
   python main.py
   ```

## Docker Setup

A `docker-compose.example.yml` is included for containerized environments.

1. Rename the example file to `docker-compose.yml`
2. Update the `DISCORD_TOKEN` environment variable with your actual bot token.
3. Ensure the mapped volume target has your `main.py` and `requirements.txt` available (the example uses `/DATA/AppData/discord_bot`).
4. Update the `OLLAMA_URL` if your Ollama instance is not available via `host.docker.internal`.
5. Run using docker compose:
   ```bash
   docker compose up -d
   ```
