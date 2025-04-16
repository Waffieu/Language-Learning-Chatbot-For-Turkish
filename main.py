import logging
import asyncio
import os
import math
import time
import concurrent.futures
from typing import Dict, List, Any, Optional, Union, Tuple

import google.generativeai as genai
import torch
from telegram import Update, Bot
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from telegram.constants import ChatAction

# Telegram message length limit (4096 characters)
MAX_MESSAGE_LENGTH = 4096

import config
from memory import Memory
from web_search import generate_search_queries, search_with_duckduckgo
from personality import create_system_prompt, format_messages_for_gemini
from language_detection import detect_language_with_gemini
from media_analysis import analyze_image, analyze_video, download_media_from_message
from deep_search import deep_search_with_progress, generate_response_with_deep_search
from time_awareness import get_time_awareness_context
from gpu_utils import gpu_manager
from word_translation import post_process_response
# Action translation no longer needed as we've removed physical action descriptions

# Configure logging with more detailed format and DEBUG level for better debugging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    filename='bot_debug.log',  # Log to a file
    filemode='a'  # Append to the log file
)

logger = logging.getLogger(__name__)

# Log startup information
logger.info("Starting Tails Bot with DuckDuckGo web search integration")
logger.info(f"Using Gemini model: {config.GEMINI_MODEL}")
logger.info(f"Short memory size: {config.SHORT_MEMORY_SIZE}, Long memory size: {config.LONG_MEMORY_SIZE}")
logger.info(f"Max search results: {config.MAX_SEARCH_RESULTS}")

# Initialize GPU manager
gpu_manager.optimize_for_inference()
logger.info(f"GPU acceleration: {gpu_manager.gpu_available}")
if gpu_manager.gpu_available:
    gpu_stats = gpu_manager.get_memory_stats()
    logger.info(f"GPU memory: {gpu_stats['allocated']:.2f}MB allocated, {gpu_stats['total']:.2f}MB total")

# Initialize thread pool for parallel processing
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# Initialize memory
memory = Memory()

# Initialize Gemini
genai.configure(api_key=config.GEMINI_API_KEY)

# User language cache
user_languages: Dict[int, str] = {}

# Response time tracking
response_times: Dict[int, Dict[str, float]] = {}

def split_long_message(text: str, max_length: int = MAX_MESSAGE_LENGTH - 100) -> List[str]:
    """
    Split a long message into chunks that respect Telegram's message length limit.
    Leaves a 100 character buffer for safety.

    Args:
        text: The message text to split
        max_length: Maximum length of each chunk (default: Telegram's limit minus 100)

    Returns:
        List of message chunks
    """
    # Log the input text length
    logger.debug(f"Splitting message with length {len(text)} into chunks of max {max_length} characters")

    # Handle empty or None text
    if not text:
        logger.error("Attempted to split empty or None text")
        return ["I couldn't generate a proper response. Please try again."]

    # If the message is already short enough, return it as is
    if len(text) <= max_length:
        logger.debug(f"Message is already short enough ({len(text)} chars), not splitting")
        return [text]

    # Split the message into chunks
    chunks = []
    current_chunk = ""

    # Split by paragraphs first (double newlines)
    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:
        # If adding this paragraph would exceed the limit, save the current chunk and start a new one
        if len(current_chunk) + len(paragraph) + 2 > max_length:  # +2 for the "\n\n"
            # If the current chunk is not empty, add it to chunks
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # If the paragraph itself is too long, split it by sentences
            if len(paragraph) > max_length:
                # Split by sentences (period followed by space or newline)
                sentences = paragraph.replace(". ", ".\n").split("\n")

                for sentence in sentences:
                    # If adding this sentence would exceed the limit, save the current chunk and start a new one
                    if len(current_chunk) + len(sentence) + 1 > max_length:  # +1 for the space
                        # If the current chunk is not empty, add it to chunks
                        if current_chunk:
                            chunks.append(current_chunk)
                            current_chunk = ""

                        # If the sentence itself is too long, split it by words
                        if len(sentence) > max_length:
                            words = sentence.split(" ")

                            for word in words:
                                # If adding this word would exceed the limit, save the current chunk and start a new one
                                if len(current_chunk) + len(word) + 1 > max_length:  # +1 for the space
                                    chunks.append(current_chunk)
                                    current_chunk = word
                                else:
                                    # Add the word to the current chunk
                                    if current_chunk:
                                        current_chunk += " " + word
                                    else:
                                        current_chunk = word
                        else:
                            # Add the sentence to the current chunk
                            current_chunk = sentence
                    else:
                        # Add the sentence to the current chunk
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
            else:
                # Add the paragraph to the current chunk
                current_chunk = paragraph
        else:
            # Add the paragraph to the current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    # Log the splitting results
    logger.info(f"Split message of {len(text)} chars into {len(chunks)} chunks")

    # Verify that chunks were created
    if not chunks:
        logger.error("No chunks were created during splitting")
        return ["I couldn't format my response properly. Please try again."]

    # Log each chunk's length for debugging
    for i, chunk in enumerate(chunks):
        logger.debug(f"Chunk {i+1}/{len(chunks)} length: {len(chunk)}")

    # Verify that no chunk exceeds the maximum length
    for i, chunk in enumerate(chunks):
        if len(chunk) > MAX_MESSAGE_LENGTH:
            logger.error(f"Chunk {i+1} exceeds Telegram's message length limit: {len(chunk)} > {MAX_MESSAGE_LENGTH}")
            # Truncate the chunk to avoid Telegram API errors
            chunks[i] = chunk[:MAX_MESSAGE_LENGTH - 100] + "\n\n[Message was too long and had to be truncated]"

    return chunks

async def keep_typing(chat_id: int, bot: Bot, cancel_event: asyncio.Event) -> None:
    """Keep sending typing action until cancel_event is set."""
    while not cancel_event.is_set():
        await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        await asyncio.sleep(4)  # Telegram typing indicator lasts about 5 seconds

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages."""
    try:
        chat_id = update.effective_chat.id
        user = update.effective_user
        message = update.message

        # Start timing the response
        start_time = time.time()

        # Track this in the response times dictionary
        if chat_id not in response_times:
            response_times[chat_id] = {}
        response_times[chat_id]['start_time'] = start_time

        # Check if this is the first message
        if chat_id not in memory.conversations or not memory.conversations[chat_id]:
            # Detect language (default to English for first message)
            detected_language = "English"
            try:
                if message.text:
                    detected_language = await asyncio.to_thread(detect_language_with_gemini, message.text)
                    user_languages[chat_id] = detected_language
            except Exception as e:
                logger.error(f"Error detecting language for first message: {e}")

            welcome_message = f"Hello, {user.first_name}! I'm Miles Prower, but everyone calls me Tails! I'm an 8-year-old fox with twin tails from the Sonic the Hedgehog universe. I'm a mechanical genius and inventor who loves to build gadgets and solve problems! I can fly by spinning my two tails like helicopter rotors, and I'm Sonic's best friend and trusted sidekick. I can search the internet to help answer your questions and provide useful information! What would you like to talk about today, friend? 😊"
            try:
                await message.reply_text(welcome_message)
                memory.add_message(chat_id, "model", welcome_message)

                # Log first response time
                end_time = time.time()
                response_times[chat_id]['first_response_time'] = end_time - start_time
                logger.info(f"First response sent in {end_time - start_time:.2f} seconds for chat {chat_id}")
            except Exception as e:
                logger.error(f"Error sending welcome message: {e}")
            return

        # Set up typing indicator that continues until response is ready
        cancel_typing = asyncio.Event()
        typing_task = asyncio.create_task(
            keep_typing(chat_id, context.bot, cancel_typing)
        )

        try:
            # Determine message type and process accordingly
            if message.text:
                # Text message
                user_message = message.text
                media_analysis = None
                media_type = "text"

                # Add user message to memory
                memory.add_message(chat_id, "user", user_message)

                # Detect language
                detected_language = await asyncio.to_thread(detect_language_with_gemini, user_message)
                user_languages[chat_id] = detected_language

            elif message.photo or message.video or (message.document and
                  message.document.mime_type and
                  (message.document.mime_type.startswith("image/") or
                   message.document.mime_type.startswith("video/"))):
                # Media message (photo or video)

                # Download the media file
                file_path, media_type = await download_media_from_message(message)

                if not file_path:
                    await message.reply_text(f"I couldn't process this media file. Please try again with a different file.")
                    if not cancel_typing.is_set():
                        cancel_typing.set()
                        await typing_task
                    return

                # Analyze the media
                if media_type == "photo":
                    media_analysis = await analyze_image(file_path)
                    user_message = f"[Image: {media_analysis['description'][:100]}]"
                elif media_type == "video":
                    media_analysis = await analyze_video(file_path)
                    user_message = f"[Video: {media_analysis['description'][:100]}]"
                else:
                    # Unsupported media type
                    await message.reply_text(f"I don't know how to process this type of file yet.")
                    if not cancel_typing.is_set():
                        cancel_typing.set()
                        await typing_task
                    return

                # Add user message to memory
                memory.add_message(chat_id, "user", user_message)

                # Use the cached language or default to English
                detected_language = user_languages.get(chat_id, "English")
            else:
                # Unsupported message type
                await message.reply_text(f"I can only understand text, images, and videos right now.")
                if not cancel_typing.is_set():
                    cancel_typing.set()
                    await typing_task
                return

            # Get chat history
            chat_history = memory.get_short_memory(chat_id)
            logger.debug(f"Retrieved {len(chat_history)} messages from short memory for chat {chat_id}")

            # Get time awareness context if enabled
            time_context = None
            if config.TIME_AWARENESS_ENABLED:
                time_context = get_time_awareness_context(chat_id)
                logger.debug(f"Time context for chat {chat_id}: {time_context['formatted_time']} (last message: {time_context['formatted_time_since']})")

            # Generate search queries
            logger.info(f"Starting web search process for message: '{user_message[:50]}...' (truncated)")

            if media_type in ("photo", "video") and media_analysis and media_analysis["search_queries"]:
                # Use search queries from media analysis
                search_queries = media_analysis["search_queries"]
                logger.info(f"Using {len(search_queries)} search queries from media analysis: {search_queries}")
            else:
                # Generate search queries from text
                logger.info(f"Generating search queries from text message")
                search_queries = await asyncio.to_thread(
                    generate_search_queries,
                    user_message,
                    chat_history
                )
                logger.info(f"Generated {len(search_queries)} search queries: {search_queries}")

            # Perform searches and collect results in parallel
            logger.info(f"Starting DuckDuckGo searches with {len(search_queries)} queries in parallel")
            search_results = []

            # Use asyncio.gather to run searches in parallel
            async def search_query(query, index):
                logger.debug(f"Executing DuckDuckGo search {index+1}/{len(search_queries)}: '{query}'")
                result = await asyncio.to_thread(search_with_duckduckgo, query)
                logger.debug(f"Search {index+1} returned {len(result['citations'])} results with {len(result['text'])} chars of text")
                return result

            # Run all searches in parallel
            search_tasks = [search_query(query, i) for i, query in enumerate(search_queries)]
            search_results = await asyncio.gather(*search_tasks)

            # Combine search results
            logger.info(f"Combining results from {len(search_results)} searches")
            combined_results = combine_search_results(search_results)
            logger.info(f"Combined search results: {len(combined_results['text'])} chars of text with {len(combined_results['citations'])} citations")

            # Generate response with search context
            response = await generate_response_with_search(
                user_message,
                chat_history,
                combined_results,
                detected_language,
                media_analysis if media_type in ("photo", "video") else None,
                time_context if config.TIME_AWARENESS_ENABLED else None,
                chat_id  # Pass chat_id for response time tracking
            )

            # Stop typing indicator
            cancel_typing.set()
            await typing_task

            # Split the response into chunks if it's too long
            response_chunks = split_long_message(response)
            logger.info(f"Sending response in {len(response_chunks)} chunks")

            # Send each chunk as a separate message
            first_chunk = True
            for chunk in response_chunks:
                if first_chunk:
                    # Send the first chunk as a reply to the original message
                    await message.reply_text(chunk)
                    first_chunk = False
                else:
                    # Send subsequent chunks as regular messages
                    await context.bot.send_message(chat_id=chat_id, text=chunk)

            # Add model response to memory (store the full response)
            memory.add_message(chat_id, "model", response)

            # Clean up temporary files if needed
            if media_type in ("photo", "video") and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error removing temporary file {file_path}: {e}")

        except Exception as e:
            # Stop typing indicator if it's running
            if not cancel_typing.is_set():
                cancel_typing.set()
                await typing_task

            logger.error(f"Error in message processing: {e}")
            try:
                error_message = f"I encountered an error. Please try again later."
                await message.reply_text(error_message)
                memory.add_message(chat_id, "model", error_message)
            except Exception as send_error:
                logger.error(f"Error sending error message: {send_error}")

    except Exception as outer_e:
        logger.error(f"Critical error in handle_message: {outer_e}")

async def should_use_web_search() -> bool:
    """
    Always return True to always use web search for every query automatically
    without requiring any commands from the user

    Returns:
        Always True to enable web search for all messages
    """
    return True  # Web search is always enabled automatically

def combine_search_results(search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine multiple search results into a single context

    Args:
        search_results: List of search result dictionaries

    Returns:
        Combined search results
    """
    # Debug: Log the number of search results to combine
    logger.debug(f"Combining {len(search_results)} search results")

    combined_text = ""
    all_citations = []

    for i, result in enumerate(search_results):
        # Debug: Log details about each result being combined
        logger.debug(f"Combining result {i+1}: {len(result['text'])} chars of text with {len(result['citations'])} citations")

        combined_text += result["text"] + "\n\n"
        all_citations.extend(result["citations"])

    # Debug: Log the final combined result
    logger.debug(f"Combined result: {len(combined_text)} chars of text with {len(all_citations)} citations")

    return {
        "text": combined_text.strip(),
        "citations": all_citations
    }

async def generate_response(
    _: str,  # user_message not used directly but kept for consistent interface
    chat_history: List[Dict[str, str]],
    language: str,
    chat_id: int = 0  # Added chat_id parameter for tracking response times
) -> str:
    """
    Generate a response using Gemini

    Args:
        user_message: The user's message
        chat_history: Recent chat history
        language: Detected language
        chat_id: The chat ID for tracking response times

    Returns:
        Generated response
    """
    start_time = time.time()

    # Create system prompt with personality
    system_prompt = create_system_prompt(language)

    # Format messages for Gemini
    prompt = format_messages_for_gemini(chat_history, system_prompt)

    try:
        # Configure Gemini with optimized settings
        model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            generation_config={
                "temperature": config.GEMINI_TEMPERATURE,
                "top_p": config.GEMINI_TOP_P,
                "top_k": config.GEMINI_TOP_K,
                "max_output_tokens": config.GEMINI_MAX_OUTPUT_TOKENS,
            },
            safety_settings=config.SAFETY_SETTINGS
        )

        # Generate response with timeout handling
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(lambda: model.generate_content(prompt).text),
                timeout=30  # 30 second timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Response generation timed out for chat {chat_id}")
            return f"I'm taking too long to respond. Let me try a simpler answer based on what I know."

        # Track response time
        if chat_id > 0:
            end_time = time.time()
            if chat_id not in response_times:
                response_times[chat_id] = {}
            response_times[chat_id]['last_response_time'] = end_time - start_time
            logger.info(f"Response generated in {end_time - start_time:.2f} seconds for chat {chat_id}")

        # Post-process response to add Turkish translations for uncommon words
        processed_response = post_process_response(response, language)
        logger.info(f"Post-processed response with Turkish translations for uncommon words")

        return processed_response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        # Default to English if language is not available
        return f"I'm having trouble processing your request. Let me try to answer based on what I know."

async def generate_response_with_search(
    _: str,  # user_message not used directly but kept for consistent interface
    chat_history: List[Dict[str, str]],
    search_results: Dict[str, Any],
    language: str,
    media_analysis: Optional[Dict[str, Any]] = None,
    time_context: Optional[Dict[str, Any]] = None,
    chat_id: int = 0  # Added chat_id parameter for tracking response times
) -> str:
    """
    Generate a response using Gemini with search results

    Args:
        user_message: The user's message
        chat_history: Recent chat history
        search_results: Combined search results
        language: Detected language
        media_analysis: Optional media analysis results
        time_context: Optional time awareness context
        chat_id: The chat ID for tracking response times

    Returns:
        Generated response
    """
    start_time = time.time()

    # Debug: Log the start of response generation
    logger.info(f"Generating response with search results in language: {language}")
    logger.debug(f"Using {len(chat_history)} messages from chat history")
    logger.debug(f"Search results: {len(search_results['text'])} chars with {len(search_results['citations'])} citations")
    if media_analysis:
        logger.debug(f"Media analysis available: {len(media_analysis['description'])} chars description")

    # Create system prompt with personality
    system_prompt = create_system_prompt(language)
    logger.debug(f"Created system prompt for language: {language}")

    # Format messages for Gemini (now returns a list)
    history_messages = format_messages_for_gemini(chat_history, system_prompt) # system_prompt is not used by the new format_messages
    logger.debug(f"Formatted message history: {len(history_messages)} messages")

    # Construct the full message list for Gemini, including system prompt and context
    full_messages_for_gemini = []

    # Add system prompt (as the first 'user' turn, a common pattern)
    # We include the personality instructions here.
    # Note: Ideally, Gemini API might have a dedicated system prompt field.
    full_messages_for_gemini.append({'role': 'user', 'parts': [system_prompt]})
    # Add a model 'OK' response to prime the conversation structure if needed
    full_messages_for_gemini.append({'role': 'model', 'parts': ["Okay, I understand. I am Miles 'Tails' Prower, ready to help!"]})

    # Add additional context before the actual history
    additional_context = "" # Initialize additional_context as an empty string

    # Add time awareness context if available
    if time_context and config.TIME_AWARENESS_ENABLED:
        logger.debug("Adding time awareness context to prompt")
        time_awareness_context = f"""
        CURRENT TIME INFORMATION:
        - Current time in Turkey: {time_context['formatted_time']}
        - Time since user's last message: {time_context['formatted_time_since']}

        IMPORTANT: You have access to this time information, but DO NOT mention the time or time-related information in your response UNLESS the user EXPLICITLY asks about the time or specifically requests time-related information. Never volunteer time information on your own.
        """
        additional_context += time_awareness_context + "\n\n"

    # Add media analysis context if available
    if media_analysis:
        logger.debug("Adding media analysis context to prompt")
        media_context = f"""
        I've analyzed the media file and here's what I found:

        Description: {media_analysis['description']}

        Please use this information along with the web search results to provide an accurate and helpful response.
        """
        additional_context += media_context + "\n\n"

    # Add search context
    logger.debug("Adding search results context to prompt")

    # Format citations for reference
    citations_info = ""
    for citation in search_results['citations']:
        citations_info += f"{citation['title']} - {citation['url']}\n"

    search_context = f"""
    I've searched the web using DuckDuckGo and found the following information that might help answer the user's question:

    {search_results['text']}

    Here are the sources I used:
    {citations_info}

    IMPORTANT: You MUST use the above web search information to provide an accurate and helpful response. DO NOT rely on your training data for factual information. Instead, use the search results above as your primary source of information.

    Please use this information to provide an accurate and helpful response while maintaining your Miles "Tails" Prower personality. Remember to:
    1. ALWAYS prioritize the web search results over your training data when answering factual questions
    2. Keep your language at A1 level if not speaking English
    3. Use simple, short sentences most of the time
    4. Occasionally use longer or more complex sentences when appropriate
    5. Speak naturally like a real person would - be conversational and friendly
    6. Use appropriate emojis occasionally to make your messages more engaging (like 😊, 👍, 🔧, ✈️, etc.)
    7. Use 1-3 emojis per message that match your personality as a young, enthusiastic fox inventor
    8. Avoid using physical action descriptions (like *tail wags* or *ears perk up*)
    9. Focus on speaking in a natural, human-like way
    10. {"ONLY provide links or sources if the user specifically asks for them or if it's directly relevant to the conversation" if config.SHOW_LINKS_ONLY_WHEN_RELEVANT else "If the user asks for links or sources, provide the relevant URLs from the citations above"}
    11. {"ONLY mention where information came from by including the URL if the user specifically asks for sources or if it's directly relevant to the conversation" if config.SHOW_LINKS_ONLY_WHEN_RELEVANT else "When providing information from sources, mention where it came from by including the URL"}
    12. DO NOT use numbered references like [1], [2], [4], [32], etc. in your response - instead, include the actual URLs if needed
    13. DO NOT mention the current time or time-related information UNLESS the user EXPLICITLY asks about the time
    14. DO NOT acknowledge how long it's been since the user's last message UNLESS the user specifically asks about it
    """
    additional_context += search_context

    # Add the combined additional context as a user message before the history
    if additional_context:
        full_messages_for_gemini.append({'role': 'user', 'parts': [additional_context]})
        # Add a model 'OK' response after context
        full_messages_for_gemini.append({'role': 'model', 'parts': ["Understood. I will use this context and the web search results to provide an accurate response based on current information, not just my training data."]})

    # Add the actual chat history
    full_messages_for_gemini.extend(history_messages)

    logger.debug(f"Created final message list with {len(full_messages_for_gemini)} turns")

    try:
        # Configure Gemini with optimized settings
        logger.debug(f"Configuring Gemini model: {config.GEMINI_MODEL}")
        model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            generation_config={
                "temperature": config.GEMINI_TEMPERATURE,
                "top_p": config.GEMINI_TOP_P,
                "top_k": config.GEMINI_TOP_K,
                "max_output_tokens": config.GEMINI_MAX_OUTPUT_TOKENS,
            },
            safety_settings=config.SAFETY_SETTINGS
        )

        # Generate response using the message list with timeout handling
        logger.info("Sending request to Gemini for final response generation using message list")
        try:
            gemini_response = await asyncio.wait_for(
                asyncio.to_thread(lambda: model.generate_content(full_messages_for_gemini)),
                timeout=45  # 45 second timeout for search-based responses
            )
        except asyncio.TimeoutError:
            logger.warning(f"Response generation with search timed out for chat {chat_id}")
            return f"I'm taking too long to process all the search results. Let me provide a simpler answer based on what I found."

        # Extract text, handling potential errors or empty responses
        try:
            response = gemini_response.text
        except ValueError as ve:
            # Handle cases where the response might be blocked or empty
            logger.error(f"Error extracting text from Gemini response: {ve}")
            logger.error(f"Gemini response prompt feedback: {gemini_response.prompt_feedback}")
            # Check for blocked content
            if gemini_response.prompt_feedback.block_reason:
                response = f"My response was blocked due to: {gemini_response.prompt_feedback.block_reason}. I cannot provide that content."
            else:
                response = "I couldn't generate a response due to an unexpected issue."
        except AttributeError:
             logger.error("Gemini response object does not contain 'text' attribute.")
             response = "I received an invalid response structure. Please try again."

        # Post-process the response to remove any numbered references
        import re
        # Remove patterns like [4], [32], [49], etc.
        response = re.sub(r'\[\d+\]', '', response)

        # Debug: Log the response length
        logger.info(f"Received response from Gemini: {len(response)} chars")
        logger.debug(f"Response preview: '{response[:100]}...' (truncated)")

        # Track response time
        if chat_id > 0:
            end_time = time.time()
            if chat_id not in response_times:
                response_times[chat_id] = {}
            response_times[chat_id]['last_response_time'] = end_time - start_time
            logger.info(f"Response with search generated in {end_time - start_time:.2f} seconds for chat {chat_id}")

            # Clear GPU cache if needed
            if gpu_manager.gpu_available and end_time - start_time > 10:  # If response took more than 10 seconds
                gpu_manager.clear_cache()

        # Post-process response to add Turkish translations for uncommon words
        processed_response = post_process_response(response, language)
        logger.info(f"Post-processed response with Turkish translations for uncommon words")

        return processed_response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        logger.exception("Detailed response generation error traceback:")
        # Default to English if language is not available
        return f"I'm having trouble processing your request. Let me try to answer based on what I know."

async def handle_deepsearch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /deepsearch command for extensive web searching."""
    try:
        chat_id = update.effective_chat.id
        message = update.message

        # Check if there's a query after the command
        if not context.args or not ' '.join(context.args).strip():
            await message.reply_text("Please provide a search query after the /deepsearch command. For example: /deepsearch quantum computing")
            return

        # Get the search query from command arguments
        search_query = ' '.join(context.args)
        logger.info(f"Received deep search command with query: '{search_query}'")

        # Add user message to memory
        memory.add_message(chat_id, "user", f"/deepsearch {search_query}")

        # Detect language with more emphasis on accuracy for deep search
        try:
            # First try to detect from the search query, specifying that it's a search query
            detected_language = await asyncio.to_thread(detect_language_with_gemini, search_query, True)

            # If we already have a language for this user and the query is very short, use the existing language
            if chat_id in user_languages and len(search_query.split()) < 3:
                logger.info(f"Query too short for reliable language detection, using existing language: {user_languages[chat_id]}")
                detected_language = user_languages[chat_id]

            # Update the language cache
            user_languages[chat_id] = detected_language
            logger.info(f"Detected language for deep search: {detected_language}")
        except Exception as e:
            # Fallback to English or existing language
            if chat_id in user_languages:
                detected_language = user_languages[chat_id]
            else:
                detected_language = "English"
            logger.error(f"Error detecting language for deep search: {e}, using {detected_language}")

        # Get chat history
        chat_history = memory.get_short_memory(chat_id)

        # Get time awareness context if enabled
        time_context = None
        if config.TIME_AWARENESS_ENABLED:
            time_context = get_time_awareness_context(chat_id)
            logger.debug(f"Time context for deep search: {time_context['formatted_time']} (last message: {time_context['formatted_time_since']})")

        # Send initial message in the appropriate language
        initial_message_text = ""
        if detected_language.lower() == "turkish":
            initial_message_text = f"'{search_query}' için derin arama başlatılıyor...\nBu işlem 1000'e kadar web sitesini arayacak ve birkaç dakika sürebilir. İlerleme durumu hakkında sizi bilgilendireceğim."
        else:
            initial_message_text = f"Starting deep search for: '{search_query}'\nThis will search up to 1000 websites and may take several minutes. I'll keep you updated on the progress."

        initial_message = await message.reply_text(initial_message_text)

        # Set up typing indicator that continues until response is ready
        cancel_typing = asyncio.Event()
        typing_task = asyncio.create_task(
            keep_typing(chat_id, context.bot, cancel_typing)
        )

        # Progress update callback
        async def progress_callback(progress_text: str):
            try:
                # Format the progress message in the appropriate language
                if detected_language.lower() == "turkish":
                    header = f"'{search_query}' için derin arama:"
                else:
                    header = f"Deep search for: '{search_query}':"

                await initial_message.edit_text(f"{header}\n\n{progress_text}")
            except Exception as e:
                logger.error(f"Error updating progress message: {e}")

        try:
            # Perform deep search with progress updates
            search_results = await deep_search_with_progress(
                search_query,
                chat_history,
                max_sites=1000,  # Search up to 1000 websites
                progress_callback=progress_callback,
                language=detected_language
            )

            # Generate response with deep search results
            logger.info(f"Generating deep search response for query: '{search_query}'")
            try:
                response = await generate_response_with_deep_search(
                    search_query,
                    chat_history,
                    search_results,
                    detected_language,
                    time_context if config.TIME_AWARENESS_ENABLED else None
                )

                if not response or len(response.strip()) == 0:
                    logger.error("Received empty response from generate_response_with_deep_search")
                    raise ValueError("Empty response from deep search")

                logger.info(f"Successfully received deep search response with length: {len(response)}")
                logger.debug(f"Response preview: '{response[:200]}...' (truncated)")
            except Exception as response_error:
                logger.error(f"Error generating deep search response: {response_error}")
                if detected_language.lower() == "turkish":
                    response = f"Arama sonuçlarını işlerken bir sorun oluştu. {search_results['stats']['unique_urls']} web sitesini {search_results['stats']['queries_used']} farklı sorgu ile aradım, ancak sonuçları düzgün şekilde işleyemedim. Lütfen tekrar deneyin."
                else:
                    response = f"I encountered a problem processing the search results. I searched {search_results['stats']['unique_urls']} websites with {search_results['stats']['queries_used']} different queries, but couldn't process the results properly. Please try again."

            # Stop typing indicator
            cancel_typing.set()
            await typing_task

            # Update the progress message with completion notice in the appropriate language
            if detected_language.lower() == "turkish":
                await initial_message.edit_text(
                    f"'{search_query}' için derin arama tamamlandı!\n\n"
                    f"{search_results['stats']['queries_used']} farklı arama sorgusu kullanarak "
                    f"{search_results['stats']['unique_urls']} benzersiz web sitesi arandı.\n"
                    f"Toplam arama süresi: {int(search_results['stats']['total_time']//60)} dakika "
                    f"{int(search_results['stats']['total_time']%60)} saniye\n\n"
                    f"Kapsamlı cevabınız hazırlanıyor..."
                )
            else:
                await initial_message.edit_text(
                    f"Deep search completed for: '{search_query}'\n\n"
                    f"Searched {search_results['stats']['unique_urls']} unique websites using "
                    f"{search_results['stats']['queries_used']} different search queries.\n"
                    f"Total search time: {int(search_results['stats']['total_time']//60)} minutes "
                    f"{int(search_results['stats']['total_time']%60)} seconds\n\n"
                    f"Preparing your comprehensive answer..."
                )

            # Split the response into chunks if it's too long
            logger.info(f"Splitting deep search response of length {len(response)} into chunks")
            response_chunks = split_long_message(response)
            logger.info(f"Split deep search response into {len(response_chunks)} chunks")

            if not response_chunks or len(response_chunks) == 0:
                logger.error("No chunks were created from the response")
                if detected_language.lower() == "turkish":
                    await message.reply_text("Yanıtı gönderirken bir sorun oluştu. Lütfen tekrar deneyin.")
                else:
                    await message.reply_text("I encountered a problem sending the response. Please try again.")
                return

            # Log the chunks for debugging
            for i, chunk in enumerate(response_chunks):
                logger.debug(f"Chunk {i+1}/{len(response_chunks)} length: {len(chunk)}")

            # Send each chunk as a separate message
            first_chunk = True
            for chunk in response_chunks:
                try:
                    if first_chunk:
                        # Send the first chunk as a reply to the original message
                        await message.reply_text(chunk)
                        logger.info(f"Sent first chunk with length {len(chunk)}")
                        first_chunk = False
                    else:
                        # Send subsequent chunks as regular messages
                        await context.bot.send_message(chat_id=chat_id, text=chunk)
                        logger.info(f"Sent additional chunk with length {len(chunk)}")

                    # Add a small delay between messages to avoid rate limiting
                    await asyncio.sleep(0.5)
                except Exception as send_error:
                    logger.error(f"Error sending message chunk: {send_error}")
                    # Try to send an error message
                    try:
                        if detected_language.lower() == "turkish":
                            await message.reply_text("Yanıtın bir kısmını gönderirken sorun oluştu. Lütfen tekrar deneyin.")
                        else:
                            await message.reply_text("I had trouble sending part of my response. Please try again.")
                    except:
                        pass
                    break

            # Add model response to memory (store the full response)
            memory.add_message(chat_id, "model", response)

        except Exception as e:
            # Stop typing indicator if it's running
            if not cancel_typing.is_set():
                cancel_typing.set()
                await typing_task

            logger.error(f"Error in deep search processing: {e}")
            if detected_language.lower() == "turkish":
                await message.reply_text(f"Derin arama sırasında bir hata oluştu. Lütfen farklı bir sorgu ile tekrar deneyin.")
            else:
                await message.reply_text(f"I encountered an error during the deep search. Please try again with a different query.")

    except Exception as outer_e:
        logger.error(f"Critical error in handle_deepsearch_command: {outer_e}")
        try:
            if detected_language and detected_language.lower() == "turkish":
                await update.message.reply_text("Derin arama ile ilgili beklenmeyen bir hata oluştu. Lütfen daha sonra tekrar deneyin.")
            else:
                await update.message.reply_text("I encountered an unexpected error with the deep search. Please try again later.")
        except:
            pass

async def error_handler(_: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and send a message to the developer."""
    logger.error(f"Exception while handling an update: {context.error}")

def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("deepsearch", handle_deepsearch_command))

    # Add message handler for text, photo, video, and document messages
    application.add_handler(MessageHandler(
        filters.TEXT | filters.PHOTO | filters.VIDEO |
        filters.Document.IMAGE | filters.Document.VIDEO,
        handle_message
    ))

    # Register error handler
    application.add_error_handler(error_handler)

    try:
        # Start the Bot
        logger.info("Starting bot polling")
        application.run_polling()
    finally:
        # Shutdown memory system properly
        logger.info("Shutting down bot")
        memory.shutdown()

        # Clean up thread pool
        thread_pool.shutdown(wait=False)

        # Log final GPU stats if available
        if gpu_manager.gpu_available:
            gpu_stats = gpu_manager.get_memory_stats()
            logger.info(f"Final GPU memory stats: {gpu_stats['allocated']:.2f}MB allocated, {gpu_stats['total']:.2f}MB total")

if __name__ == '__main__':
    main()
