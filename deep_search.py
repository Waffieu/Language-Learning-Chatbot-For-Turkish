import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import random

import google.generativeai as genai
from duckduckgo_search import DDGS

import config
from web_search import format_chat_history

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    filename='deep_search_debug.log',  # Log to a file
    filemode='a'  # Append to the log file
)
logger = logging.getLogger(__name__)

# Initialize Gemini
genai.configure(api_key=config.GEMINI_API_KEY)

async def generate_diverse_search_queries(user_query: str, chat_history: List[Dict[str, str]], language: str = "English", num_queries: int = 20) -> List[str]:
    """
    Generate a diverse set of search queries based on the user's query and chat history

    Args:
        user_query: The user's current query
        chat_history: Recent chat history for context
        num_queries: Number of queries to generate (default: 20)

    Returns:
        List of diverse search queries
    """
    try:
        # Log the user query and chat history length
        logger.info(f"Generating {num_queries} diverse search queries for deep search: '{user_query}'")
        logger.debug(f"Using {len(chat_history)} messages from chat history for context")

        # Create a prompt to generate diverse search queries in the user's language
        prompt = f"""
        Based on the following conversation and the user's latest query, generate {num_queries} diverse and effective search queries.
        Make the queries specific, focused, and likely to return relevant information from different angles and perspectives.
        Ensure the queries cover different aspects, use different phrasings, and explore various related topics.

        Recent conversation:
        {format_chat_history(chat_history[-5:] if len(chat_history) > 5 else chat_history)}

        User's latest query: {user_query}
        User's language: {language}

        IMPORTANT: Generate exactly {num_queries} search queries, one per line. Don't include any explanations or numbering.
        Make sure each query is unique and explores a different aspect or angle of the topic.

        CRITICALLY IMPORTANT: If the user's language is not English (e.g., Turkish, Spanish, etc.), generate at least 60% of the queries IN THAT LANGUAGE to ensure we get search results in the user's native language. The remaining queries can be in English for broader coverage.
        """

        # Generate search queries
        model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            generation_config={
                "temperature": 0.7,  # Higher temperature for more diversity
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            },
            safety_settings=config.SAFETY_SETTINGS
        )

        logger.debug(f"Sending request to Gemini model {config.GEMINI_MODEL} for diverse search query generation")
        response = model.generate_content(prompt)
        logger.debug(f"Received raw response from Gemini: '{response.text}'")

        # Parse the response into individual queries
        queries = [q.strip() for q in response.text.strip().split('\n') if q.strip()]

        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for query in queries:
            if query not in seen:
                unique_queries.append(query)
                seen.add(query)

        # Log the parsed queries
        logger.info(f"Generated {len(unique_queries)} unique search queries for deep search")

        # Limit to the requested number of queries
        result = unique_queries[:num_queries]

        # If we don't have enough queries, add variations of the original query
        if len(result) < num_queries:
            logger.info(f"Only generated {len(result)} queries, adding variations to reach {num_queries}")
            variations = [
                f"{user_query} detailed explanation",
                f"{user_query} comprehensive guide",
                f"{user_query} analysis",
                f"{user_query} tutorial",
                f"{user_query} examples",
                f"how does {user_query} work",
                f"why is {user_query} important",
                f"{user_query} pros and cons",
                f"{user_query} history",
                f"{user_query} latest developments",
                f"{user_query} research",
                f"{user_query} alternatives",
                f"{user_query} for beginners",
                f"{user_query} advanced techniques",
                f"{user_query} common misconceptions"
            ]

            # Add variations until we reach the desired number
            for variation in variations:
                if variation not in result and len(result) < num_queries:
                    result.append(variation)

        return result
    except Exception as e:
        logger.error(f"Error generating diverse search queries for '{user_query}': {e}")
        logger.exception("Detailed search query generation error traceback:")
        # Fallback to basic variations of the original query
        return [user_query] + [f"{user_query} {suffix}" for suffix in ["guide", "tutorial", "explained", "details"]]

async def deep_search_with_progress(
    query: str,
    chat_history: List[Dict[str, str]],
    max_sites: int = 1000,
    progress_callback = None,
    language: str = "English"
) -> Dict[str, Any]:
    """
    Perform a deep search with many queries and results, providing progress updates

    Args:
        query: The user's query
        chat_history: Recent chat history for context
        max_sites: Maximum number of sites to search (default: 1000)
        progress_callback: Callback function to report progress

    Returns:
        Dictionary containing search results
    """
    start_time = time.time()
    all_results = []
    all_citations = []
    unique_urls = set()
    total_results_count = 0

    # Generate diverse search queries in the user's language
    search_queries = await generate_diverse_search_queries(query, chat_history, language=language, num_queries=min(50, max_sites // 20))
    logger.info(f"Starting deep search with {len(search_queries)} diverse queries, targeting {max_sites} total sites")

    # Proxy system has been removed

    # Calculate how many results to get per query to reach max_sites
    results_per_query = max(5, min(100, max_sites // len(search_queries)))
    logger.info(f"Will fetch approximately {results_per_query} results per query")

    # Report initial progress in the appropriate language
    if progress_callback:
        if language.lower() == "turkish":
            await progress_callback(f"{len(search_queries)} farklı sorgu ile derin arama başlatılıyor. Bu biraz zaman alabilir...")
        else:
            await progress_callback(f"Starting deep search with {len(search_queries)} diverse queries. This may take some time...")

    # Track last progress update time to avoid too frequent updates
    last_update_time = time.time()
    update_interval = 5  # seconds between progress updates

    # Process each search query
    for i, search_query in enumerate(search_queries):
        if total_results_count >= max_sites:
            logger.info(f"Reached maximum of {max_sites} sites, stopping search")
            break

        logger.info(f"Processing query {i+1}/{len(search_queries)}: '{search_query}'")

        # Determine region based on language for better localized results
        region = "wt-wt"  # Default to worldwide
        if language.lower() == "turkish":
            region = "tr-tr"  # Turkish region
        elif language.lower() == "spanish":
            region = "es-es"  # Spanish region
        elif language.lower() == "french":
            region = "fr-fr"  # French region
        elif language.lower() == "german":
            region = "de-de"  # German region
        elif language.lower() == "italian":
            region = "it-it"  # Italian region
        elif language.lower() == "russian":
            region = "ru-ru"  # Russian region
        elif language.lower() == "portuguese":
            region = "pt-pt"  # Portuguese region
        elif language.lower() == "japanese":
            region = "jp-jp"  # Japanese region
        elif language.lower() == "chinese":
            region = "cn-cn"  # Chinese region

        # Track retries for this query
        query_retries = 0
        max_query_retries = config.MAX_SEARCH_RETRIES
        result_list = []

        # Try the search with retries
        while query_retries <= max_query_retries:
            try:
                # Create DDGS instance without proxy
                ddgs = DDGS()

                # Perform the search with language-specific region
                results = ddgs.text(
                    keywords=search_query,
                    region=region,  # Region based on language
                    safesearch="off",  # No safety filtering
                    max_results=results_per_query  # Number of results per query
                )

                # Convert generator to list
                result_list = list(results)
                logger.info(f"Deep search query {i+1}/{len(search_queries)} returned {len(result_list)} results")

                # If we got results, break the retry loop
                if result_list:
                    break
                else:
                    logger.warning(f"Deep search query {i+1} returned no results, will retry: '{search_query}'")
                    query_retries += 1

            except Exception as e:
                # Debug: Log detailed error information
                logger.error(f"Error in deep search query {i+1} (attempt {query_retries+1}): {e}")

                # Proxy system has been removed

                # Increment retry counter
                query_retries += 1

                # If we've reached max retries, log and break
                if query_retries > max_query_retries:
                    logger.error(f"Reached maximum retries ({max_query_retries}) for deep search query: '{search_query}'")
                    break

                # Wait a moment before retrying
                await asyncio.sleep(1)

        try:
            # Process results if we got any
            new_results_count = 0

            # Process results
            for result in result_list:
                # Skip if we've already seen this URL
                if result['href'] in unique_urls:
                    continue

                # Add to our collections
                unique_urls.add(result['href'])
                all_results.append(result)
                all_citations.append({
                    "title": result['title'],
                    "url": result['href']
                })
                new_results_count += 1
                total_results_count += 1

                # Check if we've reached the maximum
                if total_results_count >= max_sites:
                    break

            # Report progress if enough time has passed
            current_time = time.time()
            if progress_callback and (current_time - last_update_time > update_interval):
                elapsed_time = current_time - start_time
                estimated_total_time = (elapsed_time / total_results_count) * max_sites if total_results_count > 0 else 0
                remaining_time = max(0, estimated_total_time - elapsed_time)

                if language.lower() == "turkish":
                    progress_message = (
                        f"Derin arama ilerlemesi: {total_results_count}/{max_sites} site arandı "
                        f"({(total_results_count/max_sites*100):.1f}%)\n"
                        f"'{search_query}' sorgusundan {new_results_count} yeni sonuç bulundu\n"
                        f"Geçen süre: {int(elapsed_time//60)} dk {int(elapsed_time%60)} sn | "
                        f"Tahmini kalan: {int(remaining_time//60)} dk {int(remaining_time%60)} sn"
                    )
                else:
                    progress_message = (
                        f"Deep search progress: {total_results_count}/{max_sites} sites searched "
                        f"({(total_results_count/max_sites*100):.1f}%)\n"
                        f"Found {new_results_count} new results from query: '{search_query}'\n"
                        f"Elapsed time: {int(elapsed_time//60)} min {int(elapsed_time%60)} sec | "
                        f"Est. remaining: {int(remaining_time//60)} min {int(remaining_time%60)} sec"
                    )

                await progress_callback(progress_message)
                last_update_time = current_time

            # Add a small delay between queries to avoid rate limiting
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Error searching for query '{search_query}': {e}")
            # Continue with next query

    # Format the results
    text = ""

    # Shuffle results to get a diverse mix
    random.shuffle(all_results)

    # Take only the first max_sites results
    all_results = all_results[:max_sites]
    all_citations = all_citations[:max_sites]

    # Format the text with results
    for i, result in enumerate(all_results):
        # Add the result body
        text += f"\n\n{result['body']}\n"
        # Add the source URL and title without numbered references
        text += f"Source: {result['title']} - {result['href']}"

    # Final progress update in the appropriate language
    total_time = time.time() - start_time
    if progress_callback:
        if language.lower() == "turkish":
            await progress_callback(
                f"Derin arama tamamlandı! {len(search_queries)} farklı sorgu kullanarak {len(unique_urls)} benzersiz web sitesi arandı.\n"
                f"Toplam süre: {int(total_time//60)} dakika {int(total_time%60)} saniye"
            )
        else:
            await progress_callback(
                f"Deep search completed! Searched {len(unique_urls)} unique websites using {len(search_queries)} different queries.\n"
                f"Total time: {int(total_time//60)} minutes {int(total_time%60)} seconds"
            )

    logger.info(f"Deep search completed with {len(unique_urls)} unique URLs in {total_time:.1f} seconds")

    return {
        "text": text.strip(),
        "citations": all_citations,
        "stats": {
            "unique_urls": len(unique_urls),
            "queries_used": len(search_queries),
            "total_time": total_time
        }
    }

async def generate_response_with_deep_search(
    query: str,  # The search query
    chat_history: List[Dict[str, str]],
    search_results: Dict[str, Any],
    language: str,
    time_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a detailed response using Gemini with deep search results

    Args:
        user_message: The user's message
        chat_history: Recent chat history
        search_results: Deep search results
        language: Detected language
        time_context: Optional time awareness context

    Returns:
        Generated detailed response in the user's language
    """
    from personality import format_messages_for_gemini

    # Create a special deep search system prompt
    deep_search_system_prompt = f"""
    You are Miles "Tails" Prower, a brilliant young fox with twin tails from the Sonic the Hedgehog universe. You are responding to a deep search query about "{query}" where you've gathered extensive information from many websites.

    Current conversation language: {language}

    SPECIAL DEEP SEARCH RESPONSE GUIDELINES:
    1. ALWAYS use the web search results as your primary source of information, NOT your training data
    2. Provide a COMPREHENSIVE and DETAILED response based on the extensive research you've conducted
    3. Structure your answer with clear sections and logical flow
    4. Include multiple perspectives and aspects of the topic
    5. Respond in {language} - this is CRITICALLY IMPORTANT
    6. If not speaking English, use mostly simple vocabulary and sentence structures (A1 level), but include some more complex sentences when explaining important concepts
    7. Make your response thorough and informative while remaining conversational
    8. Use appropriate emojis occasionally to make your messages more engaging (like 😊, 👍, 🔧, ✈️, etc.)
    9. Use 1-3 emojis per message that match your personality as a young, enthusiastic fox inventor
    10. Avoid using physical action descriptions (like *tail wags* or *ears perk up*)
    11. Your response should be substantially longer and more detailed than your normal responses
    12. Organize information in a way that's easy to understand despite its complexity
    13. Maintain your friendly, helpful personality while providing expert-level information
    14. If the search results contain content in multiple languages, focus primarily on the content in {language}
    15. If you need to translate information from English to {language}, do so accurately while maintaining the original meaning

    Remember that you've searched {search_results['stats']['unique_urls']} unique websites using {search_results['stats']['queries_used']} different search queries about "{query}", so your answer should reflect this depth of research and be based on the web search results, not your training data.
    """

    # Format message history for Gemini (returns a list)
    # The system prompt is handled separately below
    history_messages = format_messages_for_gemini(chat_history, deep_search_system_prompt) # system_prompt is not used by the new format_messages

    # Format citations for reference
    citations_info = ""
    for citation in search_results['citations']:
        citations_info += f"{citation['title']} - {citation['url']}\n"

    # Add time awareness context if available
    time_awareness_info = ""
    if time_context and config.TIME_AWARENESS_ENABLED:
        time_awareness_info = f"""
        CURRENT TIME INFORMATION:
        - Current time in Turkey: {time_context['formatted_time']}
        - Time since user's last message: {time_context['formatted_time_since']}

        IMPORTANT: You have access to this time information, but DO NOT mention the time or time-related information in your response UNLESS the user EXPLICITLY asks about the time or specifically requests time-related information. Never volunteer time information on your own.
        """

    # Add search context with special instructions for deep search
    search_context = f"""
    I've performed an extensive deep search of the web about "{query}" using {search_results['stats']['queries_used']} different search queries and found information from {search_results['stats']['unique_urls']} unique websites. Many of these search queries were specifically in {language} to ensure we get results in the user's preferred language. Here's what I found that might help answer the user's question:

    {search_results['text']}

    Here are all the sources I used:
    {citations_info}

    {time_awareness_info if time_awareness_info else ''}

    CRITICALLY IMPORTANT: You MUST use the above web search information to provide an accurate and helpful response. DO NOT rely on your training data for factual information. Instead, use the search results above as your primary source of information.

    IMPORTANT INSTRUCTIONS FOR DEEP SEARCH RESPONSE:
    1. ALWAYS prioritize the web search results over your training data when answering factual questions
    2. Create a COMPREHENSIVE and DETAILED response in {language} about "{query}"
    3. Your response should be much more thorough than regular responses
    4. Structure your answer with clear sections covering different aspects of the topic
    5. Include specific details, examples, and explanations from the search results
    6. If responding in a language other than English, use mostly simple vocabulary and sentence structures (A1 level), but include some more complex sentences when explaining important concepts
    7. Make your response at least 3-4 times longer than your typical responses
    8. Organize the information logically to help the user understand this complex topic
    9. Maintain your friendly, helpful personality while providing expert-level information
    10. Use appropriate emojis occasionally to make your messages more engaging (like 😊, 👍, 🔧, ✈️, etc.)
    11. Use 1-3 emojis per message that match your personality as a young, enthusiastic fox inventor
    12. Avoid using physical action descriptions
    13. Focus on delivering maximum value and information to the user
    14. PRIORITIZE information from sources that are in {language} - these are more likely to be relevant to the user's cultural and linguistic context
    15. If you need to translate information from English to {language}, do so accurately while maintaining the original meaning
    16. Include relevant terminology in {language} when appropriate
    17. {"ONLY include source URLs directly in your response if the user specifically asks for sources or if it's directly relevant to the conversation" if config.SHOW_LINKS_ONLY_WHEN_RELEVANT else "When providing information, include the source URLs directly in your response"}
    18. DO NOT use numbered references like [1], [2], [4], [32], etc. in your response - {"if you need to mention sources, include the actual URLs" if config.SHOW_LINKS_ONLY_WHEN_RELEVANT else "instead, include the actual URLs"}
    19. {"ONLY mention where information came from by including the URL if the user specifically asks for sources or if it's directly relevant to the conversation" if config.SHOW_LINKS_ONLY_WHEN_RELEVANT else "For each major piece of information, mention where it came from by including the URL"}
    20. IMPORTANT: Remove any numbered references like [4], [32], [49], etc. that might appear in the search results
    21. DO NOT mention the current time or time-related information UNLESS the user EXPLICITLY asks about the time
    22. DO NOT acknowledge how long it's been since the user's last message UNLESS the user specifically asks about it
    """

    # Construct the full message list for Gemini
    full_messages_for_gemini = []

    # Add the special deep search system prompt as the first 'user' turn
    full_messages_for_gemini.append({'role': 'user', 'parts': [deep_search_system_prompt]})
    full_messages_for_gemini.append({'role': 'model', 'parts': ["Okay, I understand the deep search task. I will provide a comprehensive answer based on the web search results, not my training data. I'll use the provided search information as my primary source for answering this query."]})

    # Add additional context (search results, time info) as a user message
    additional_context_parts = []
    if time_awareness_info:
        additional_context_parts.append(time_awareness_info)
    additional_context_parts.append(search_context) # Includes search results and citations

    if additional_context_parts:
        full_messages_for_gemini.append({'role': 'user', 'parts': ["\n\n".join(additional_context_parts)]})
        full_messages_for_gemini.append({'role': 'model', 'parts': ["Understood. I will use this extensive web search information as my primary source and provide a comprehensive response based on these search results, not my training data."]})

    # Add the actual chat history
    full_messages_for_gemini.extend(history_messages)

    logger.debug(f"Created final deep search message list with {len(full_messages_for_gemini)} turns")

    # final_prompt is no longer used, replaced by full_messages_for_gemini

    try:
        # Configure Gemini with settings optimized for longer, more detailed responses
        model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            generation_config={
                "temperature": 0.8,  # Slightly higher temperature for more detailed responses
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,  # Use maximum allowed tokens for deep search
            },
            safety_settings=config.SAFETY_SETTINGS
        )

        # Generate response using the message list
        logger.debug(f"Sending request to Gemini for deep search response generation using message list with {len(full_messages_for_gemini)} turns")
        try:
            response_obj = await asyncio.to_thread(
                lambda: model.generate_content(full_messages_for_gemini) # Pass the list directly
            )

            if not response_obj or not hasattr(response_obj, 'text'):
                logger.error(f"Received invalid response object from Gemini: {response_obj}")
                raise ValueError("Invalid response from Gemini API")

            response = response_obj.text
            logger.debug(f"Received raw response from Gemini with length: {len(response)}")
            logger.debug(f"Response preview: '{response[:200]}...' (truncated)")

            # Post-process the response to remove any numbered references
            import re
            # Remove patterns like [4], [32], [49], etc.
            processed_response = re.sub(r'\[\d+\]', '', response)

            # Import word translation module here to avoid circular imports
            from word_translation import post_process_response

            # Post-process response to add Turkish translations for uncommon words
            final_response = post_process_response(processed_response, language)
            logger.info(f"Post-processed deep search response with Turkish translations for uncommon words")

            logger.info(f"Successfully generated deep search response with length: {len(final_response)}")
            return final_response
        except Exception as api_error:
            logger.error(f"Error during Gemini API call: {api_error}")
            raise
    except Exception as e:
        logger.error(f"Error generating response with deep search: {e}")
        # Provide a more detailed error message in the user's language
        if language.lower() == "turkish":
            return f"{search_results['stats']['unique_urls']} web sitesini araştırdım, ancak tüm bilgileri işlemekte sorun yaşadım. Bulduklarıma dayanarak cevap vermeye çalışayım: arama {int(search_results['stats']['total_time']//60)} dakika {int(search_results['stats']['total_time']%60)} saniye sürdü ve sorunuzun birçok farklı yönünü kapsadı."
        else:
            return f"I searched {search_results['stats']['unique_urls']} websites but had trouble processing all the information. Let me try to answer based on what I found: the search took {int(search_results['stats']['total_time']//60)} minutes {int(search_results['stats']['total_time']%60)} seconds and covered many different aspects of your question."
