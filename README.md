# Miles "Tails" Prower Telegram Bot

A Telegram chatbot based on Miles "Tails" Prower from the Sonic the Hedgehog universe, powered by Google's Gemini AI models with a hybrid system using both `gemini-2.0-flash-thinking-exp-01-21` and `gemini-2.0-flash-lite`.

## Features

- **Tails Personality**: The bot embodies Miles "Tails" Prower, the twin-tailed fox genius inventor and Sonic's best friend from the Sonic the Hedgehog universe
- **Typing Indicator**: Shows typing animation while generating responses
- **Hybrid Model System**:
  - Uses specialized Gemini model configurations for different tasks
  - Main conversation: `gemini-2.0-flash-thinking-exp-01-21` with balanced temperature (0.7)
  - Word translation: `gemini-2.0-flash-lite` with low temperature (0.1) for accuracy
  - Search query generation: `gemini-2.0-flash-lite` with moderate temperature (0.2)
  - Language detection: `gemini-2.0-flash-lite` with low temperature (0.1) for precision
  - Each model is optimized for its specific task to improve overall performance
- **Web Search Capabilities**:
  - **Automatic Web Search**: Automatically searches the web for every query to provide accurate information
  - **Deep Search Command**: Use `/deepsearch` to search up to 1000 websites with diverse queries for comprehensive answers

- **Persistent Memory System**:
  - Short-term memory (25 messages) for immediate context
  - Long-term memory (100 messages) for each user
  - Memories persist between bot restarts
  - Each user has their own personalized memory file
- **Time Awareness**:
  - Understands the current time in Turkey
  - Recognizes time of day (morning, afternoon, evening, night)
  - Tracks how long it's been since the user's last message
  - Naturally references time information in conversations
- **Language Adaptation**:
  - Automatically detects and responds in the user's language
  - Provides A1-level language responses for non-native speakers
  - Translates uncommon words to Turkish for language learners
  - Groups all word translations at the end of messages in a "wort schatz" section
  - Adds friendly emojis to make responses more engaging
- **Natural Responses**: Provides conversational, easy-to-understand responses

## Setup

### Prerequisites

- Python 3.9+
- A Telegram Bot Token (from [BotFather](https://t.me/botfather))
- A Google Gemini API Key (from [Google AI Studio](https://aistudio.google.com/))

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/tails-telegram-bot.git
   cd tails-telegram-bot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file based on the `.env.example` template:
   ```
   cp .env.example .env
   ```

4. Edit the `.env` file with your API keys and configuration:
   ```
   # Telegram Bot Token (get from BotFather)
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

   # Google Gemini API Key
   GEMINI_API_KEY=your_gemini_api_key_here

   # Memory settings
   SHORT_MEMORY_SIZE=25
   LONG_MEMORY_SIZE=100
   MEMORY_DIR=user_memories

   # Web search settings
   MAX_SEARCH_RESULTS=5

   # Maximum number of retries for DuckDuckGo searches
   MAX_SEARCH_RETRIES=5
   ```

### Running the Bot

Run the bot with:
```
python main.py
```

## Usage

Once the bot is running, you can interact with it on Telegram by simply sending messages or using commands.

### Regular Chat

Simply send messages to the bot and it will:

- Automatically respond in your language
- Search the web for relevant information for every query
- Remember your entire conversation history
- Show a typing indicator while generating responses

### Commands

- `/deepsearch [query]` - Performs an extensive search across up to 1000 websites using multiple search queries. This provides much more comprehensive information than regular searches. For example: `/deepsearch quantum computing advancements`
  - The bot will continuously update you on the search progress
  - Searches can take several minutes to complete depending on the complexity of the query
  - Results are much more detailed and comprehensive than regular searches
  - Searches are performed in the user's language - if you search in Turkish, the bot will prioritize Turkish language results
  - The bot generates search queries in your language to ensure relevant, localized results
  - Responses are provided in the same language as your search query

### Word Translation Feature

When chatting in languages other than Turkish, the bot automatically translates uncommon words to Turkish to help language learners. All translations are grouped at the end of the message in a "wort schatz" section.

Example:

```
Heldmaschine ist eine deutsche Band!

Sie machen Musik, die NDH heißt.

Ich habe im Internet nachgeschaut.

*   Sie wurde 2008 gegründet. 👍
*   Sie kommt aus Koblenz. Magst du ihre Musik? 🦊🔧

wort schatz
heldmaschine = kahraman makinesi
heißt = denir
internet = internet
gegründet = kuruldu 😊
```

This feature helps Turkish speakers learn new vocabulary in other languages while maintaining a clean, readable message format.

## Customization

- Adjust Tails' personality in `personality.py`
- Modify memory settings in `.env`
- Configure the hybrid model system in `config.py`:
  ```python
  # Main conversation model
  GEMINI_MODEL = "gemini-2.0-flash-thinking-exp-01-21"
  GEMINI_TEMPERATURE = 0.7
  GEMINI_TOP_P = 0.95
  GEMINI_TOP_K = 40
  GEMINI_MAX_OUTPUT_TOKENS = 2048

  # Word translation model
  WORD_TRANSLATION_MODEL = "gemini-2.0-flash-lite"
  WORD_TRANSLATION_TEMPERATURE = 0.1
  WORD_TRANSLATION_TOP_P = 0.95
  WORD_TRANSLATION_TOP_K = 40
  WORD_TRANSLATION_MAX_OUTPUT_TOKENS = 200

  # Search query generation model
  SEARCH_QUERY_MODEL = "gemini-2.0-flash-lite"
  SEARCH_QUERY_TEMPERATURE = 0.2
  SEARCH_QUERY_TOP_P = 0.95
  SEARCH_QUERY_TOP_K = 40
  SEARCH_QUERY_MAX_OUTPUT_TOKENS = 256

  # Language detection model
  LANGUAGE_DETECTION_MODEL = "gemini-2.0-flash-lite"
  LANGUAGE_DETECTION_TEMPERATURE = 0.1
  LANGUAGE_DETECTION_TOP_P = 0.95
  LANGUAGE_DETECTION_TOP_K = 40
  LANGUAGE_DETECTION_MAX_OUTPUT_TOKENS = 10
  ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Recent Updates

### April 2025
- **Hybrid Model System Implementation**: Specialized Gemini model configurations for different tasks:
  - Main conversation model (`gemini-2.0-flash-thinking-exp-01-21`): Optimized for natural, engaging responses
  - Word translation model (`gemini-2.0-flash-lite`): Configured for high accuracy translations
  - Search query generation model (`gemini-2.0-flash-lite`): Tuned for relevant search queries
  - Language detection model (`gemini-2.0-flash-lite`): Optimized for precise language identification
- **Improved Word Translation Format**: All word translations ("wort schatz") are now grouped together at the end of messages instead of appearing after each sentence
- **Enhanced Translation Detection**: Better identification of uncommon words that need translation for language learners
- **Optimized Memory System**: Improved performance with GPU acceleration for faster responses
- **Removed Action Descriptions**: Physical action descriptions have been removed for more natural conversation
- **Time Awareness Improvements**: Time information is now only shown when specifically requested
- **Deep Search Enhancements**: Better progress updates and language-specific results

## Acknowledgments

- Inspired by the character Miles "Tails" Prower from the Sonic the Hedgehog franchise by SEGA
- Powered by Google's Gemini AI
- Built with python-telegram-bot
