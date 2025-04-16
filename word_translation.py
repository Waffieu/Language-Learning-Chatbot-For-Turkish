import logging
import google.generativeai as genai
import config
import re
from typing import List, Dict

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Gemini
genai.configure(api_key=config.GEMINI_API_KEY)

# Cache for translated words to avoid repeated API calls
word_translation_cache = {}

def identify_uncommon_words(text: str, language: str) -> List[str]:
    """
    Identify truly uncommon words in the given text based on language level A1
    Only selects words that would be genuinely difficult for beginners

    Args:
        text: The text to analyze
        language: The language of the text

    Returns:
        List of uncommon words
    """
    # Skip if the text is in Turkish (no need to translate to Turkish)
    if language.lower() == "turkish":
        return []

    # Remove punctuation and split into words
    words = re.findall(r'\b\w+\b', text.lower())

    # Remove duplicates while preserving order
    unique_words = []
    seen = set()
    for word in words:
        if word not in seen and len(word) > 3:  # Only consider words longer than 3 characters
            seen.add(word)
            unique_words.append(word)

    # If no words to analyze, return empty list
    if not unique_words:
        return []

    # Use a simple heuristic approach as fallback
    # These are common words in various languages that are likely to be known at A1 level
    common_words = {
        "german": [
            "ich", "du", "er", "sie", "es", "wir", "ihr", "sie", "und", "oder", "aber", "wenn", "dann",
            "bin", "ist", "sind", "war", "waren", "haben", "hat", "hatte", "gehen", "kommen", "machen",
            "gut", "schlecht", "ja", "nein", "bitte", "danke", "hallo", "tschüss", "auch", "schon",
            "noch", "jetzt", "hier", "dort", "heute", "morgen", "gestern", "immer", "nie", "oft",
            "manchmal", "viel", "wenig", "groß", "klein", "alt", "jung", "neu", "tag", "nacht",
            "woche", "monat", "jahr", "zeit", "uhr", "minute", "stunde", "wasser", "essen", "trinken",
            "sprechen", "sagen", "fragen", "antworten", "verstehen", "wissen", "kennen", "lernen",
            "leben", "lieben", "mögen", "wollen", "können", "müssen", "sollen", "dürfen", "freund",
            "familie", "mutter", "vater", "bruder", "schwester", "kind", "mann", "frau", "mensch",
            "haus", "wohnung", "stadt", "land", "straße", "weg", "auto", "bus", "zug", "flugzeug",
            "schule", "arbeit", "beruf", "geld", "preis", "farbe", "rot", "blau", "grün", "gelb",
            "schwarz", "weiß", "musik", "buch", "film", "spiel", "sport", "fußball", "wetter", "sonne",
            "regen", "schnee", "wind", "warm", "kalt", "schön", "hässlich", "leicht", "schwer", "schnell",
            "langsam", "früh", "spät", "oben", "unten", "links", "rechts", "vorne", "hinten", "in", "an",
            "auf", "unter", "über", "vor", "nach", "mit", "ohne", "für", "gegen", "weil", "dass", "ob",
            "wie", "was", "wer", "wo", "wann", "warum", "eins", "zwei", "drei", "vier", "fünf", "sechs",
            "sieben", "acht", "neun", "zehn", "hundert", "tausend", "million", "erste", "zweite", "letzte",
            "nächste", "alle", "viele", "einige", "keine", "jeder", "dieser", "jener", "mein", "dein", "sein",
            "ihr", "unser", "euer", "der", "die", "das", "ein", "eine", "kein", "keine", "mehr", "weniger",
            "sehr", "zu", "nicht", "nur", "schon", "noch", "wieder", "immer", "nie", "vielleicht", "sicher",
            "natürlich", "genau", "wirklich", "leider", "hoffentlich", "bitte", "danke", "entschuldigung",
            "tut", "leid", "gern", "geschehen", "willkommen", "tschüss", "auf", "wiedersehen", "bis", "später",
            "morgen", "bald", "guten", "tag", "abend", "nacht", "morgen", "woche", "ende", "schön", "toll",
            "super", "prima", "gut", "schlecht", "richtig", "falsch", "wichtig", "interessant", "langweilig",
            "einfach", "schwierig", "möglich", "unmöglich", "gleich", "anders", "ähnlich", "verschieden"
        ],
        "english": [
            "i", "you", "he", "she", "it", "we", "they", "and", "or", "but", "if", "then", "am", "is",
            "are", "was", "were", "have", "has", "had", "go", "come", "make", "good", "bad", "yes", "no",
            "please", "thank", "hello", "bye", "also", "already", "still", "now", "here", "there", "today",
            "tomorrow", "yesterday", "always", "never", "often", "sometimes", "much", "little", "big",
            "small", "old", "young", "new", "day", "night", "week", "month", "year", "time", "clock",
            "minute", "hour", "water", "eat", "drink", "speak", "say", "ask", "answer", "understand",
            "know", "learn", "live", "love", "like", "want", "can", "must", "should", "may", "friend",
            "family", "mother", "father", "brother", "sister", "child", "man", "woman", "person", "house",
            "apartment", "city", "country", "street", "way", "car", "bus", "train", "plane", "school",
            "work", "job", "money", "price", "color", "red", "blue", "green", "yellow", "black", "white",
            "music", "book", "movie", "game", "sport", "football", "weather", "sun", "rain", "snow",
            "wind", "warm", "cold", "beautiful", "ugly", "easy", "difficult", "fast", "slow", "early",
            "late", "up", "down", "left", "right", "front", "back", "in", "at", "on", "under", "over",
            "before", "after", "with", "without", "for", "against", "because", "that", "if", "how",
            "what", "who", "where", "when", "why", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "hundred", "thousand", "million", "first", "second", "last", "next",
            "all", "many", "some", "none", "every", "this", "that", "my", "your", "his", "her", "our",
            "their", "the", "a", "an", "no", "more", "less", "very", "too", "not", "only", "already",
            "still", "again", "always", "never", "maybe", "sure", "of course", "exactly", "really",
            "unfortunately", "hopefully", "please", "thank you", "sorry", "you're welcome", "goodbye",
            "see you", "later", "tomorrow", "soon", "good", "evening", "night", "morning", "weekend",
            "nice", "great", "super", "excellent", "good", "bad", "right", "wrong", "important",
            "interesting", "boring", "simple", "difficult", "possible", "impossible", "same", "different",
            "similar", "various"
        ],
        "french": [
            "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "et", "ou", "mais", "si", "suis",
            "est", "sommes", "sont", "étais", "étaient", "avoir", "aller", "venir", "faire", "bon",
            "mauvais", "oui", "non", "s'il", "merci", "bonjour", "au revoir", "aussi", "déjà", "encore",
            "maintenant", "ici", "là", "aujourd'hui", "demain", "hier", "toujours", "jamais", "souvent",
            "parfois", "beaucoup", "peu", "grand", "petit", "vieux", "jeune", "nouveau", "jour", "nuit",
            "semaine", "mois", "année", "temps", "horloge", "minute", "heure", "eau", "manger", "boire",
            "parler", "dire", "demander", "répondre", "comprendre", "savoir", "connaître", "apprendre",
            "vivre", "aimer", "vouloir", "pouvoir", "devoir", "ami", "famille", "mère", "père", "frère",
            "sœur", "enfant", "homme", "femme", "personne", "maison", "appartement", "ville", "pays",
            "rue", "chemin", "voiture", "bus", "train", "avion", "école", "travail", "métier", "argent",
            "prix", "couleur", "rouge", "bleu", "vert", "jaune", "noir", "blanc", "musique", "livre",
            "film", "jeu", "sport", "football", "temps", "soleil", "pluie", "neige", "vent", "chaud",
            "froid", "beau", "laid", "facile", "difficile", "rapide", "lent", "tôt", "tard", "haut",
            "bas", "gauche", "droite", "avant", "arrière", "dans", "à", "sur", "sous", "au-dessus",
            "avant", "après", "avec", "sans", "pour", "contre", "parce que", "que", "si", "comment",
            "quoi", "qui", "où", "quand", "pourquoi", "un", "deux", "trois", "quatre", "cinq", "six",
            "sept", "huit", "neuf", "dix", "cent", "mille", "million", "premier", "deuxième", "dernier",
            "prochain", "tout", "beaucoup", "quelques", "aucun", "chaque", "ce", "cette", "mon", "ton",
            "son", "notre", "votre", "leur", "le", "la", "les", "un", "une", "des", "plus", "moins",
            "très", "trop", "ne pas", "seulement", "déjà", "encore", "toujours", "jamais", "peut-être",
            "sûr", "bien sûr", "exactement", "vraiment", "malheureusement", "j'espère", "s'il vous plaît",
            "merci", "désolé", "de rien", "au revoir", "à bientôt", "à demain", "bientôt", "bonjour",
            "bonsoir", "bonne nuit", "bon matin", "week-end", "bien", "super", "excellent", "bon",
            "mauvais", "correct", "faux", "important", "intéressant", "ennuyeux", "simple", "difficile",
            "possible", "impossible", "même", "différent", "similaire", "divers"
        ],
        "spanish": [
            "yo", "tú", "él", "ella", "nosotros", "vosotros", "ellos", "ellas", "y", "o", "pero", "si",
            "soy", "es", "somos", "son", "era", "eran", "tener", "ir", "venir", "hacer", "bueno", "malo",
            "sí", "no", "por favor", "gracias", "hola", "adiós", "también", "ya", "todavía", "ahora",
            "aquí", "allí", "hoy", "mañana", "ayer", "siempre", "nunca", "a menudo", "a veces", "mucho",
            "poco", "grande", "pequeño", "viejo", "joven", "nuevo", "día", "noche", "semana", "mes",
            "año", "tiempo", "reloj", "minuto", "hora", "agua", "comer", "beber", "hablar", "decir",
            "preguntar", "responder", "entender", "saber", "conocer", "aprender", "vivir", "amar",
            "gustar", "querer", "poder", "deber", "amigo", "familia", "madre", "padre", "hermano",
            "hermana", "niño", "hombre", "mujer", "persona", "casa", "apartamento", "ciudad", "país",
            "calle", "camino", "coche", "autobús", "tren", "avión", "escuela", "trabajo", "profesión",
            "dinero", "precio", "color", "rojo", "azul", "verde", "amarillo", "negro", "blanco", "música",
            "libro", "película", "juego", "deporte", "fútbol", "tiempo", "sol", "lluvia", "nieve",
            "viento", "caliente", "frío", "bonito", "feo", "fácil", "difícil", "rápido", "lento",
            "temprano", "tarde", "arriba", "abajo", "izquierda", "derecha", "delante", "detrás", "en",
            "a", "sobre", "bajo", "encima", "antes", "después", "con", "sin", "para", "contra", "porque",
            "que", "si", "cómo", "qué", "quién", "dónde", "cuándo", "por qué", "uno", "dos", "tres",
            "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "cien", "mil", "millón",
            "primero", "segundo", "último", "próximo", "todo", "muchos", "algunos", "ninguno", "cada",
            "este", "ese", "mi", "tu", "su", "nuestro", "vuestro", "su", "el", "la", "los", "las", "un",
            "una", "unos", "unas", "más", "menos", "muy", "demasiado", "no", "solo", "ya", "todavía",
            "otra vez", "siempre", "nunca", "quizás", "seguro", "por supuesto", "exactamente", "realmente",
            "desafortunadamente", "espero", "por favor", "gracias", "lo siento", "de nada", "adiós",
            "hasta luego", "hasta mañana", "pronto", "buenos días", "buenas tardes", "buenas noches",
            "buen día", "fin de semana", "bien", "genial", "excelente", "bueno", "malo", "correcto",
            "incorrecto", "importante", "interesante", "aburrido", "simple", "difícil", "posible",
            "imposible", "igual", "diferente", "similar", "varios"
        ]
    }

    # Use Gemini to identify uncommon words
    try:
        prompt = f"""
        Analyze the following list of words in {language} and identify ONLY the truly uncommon or difficult words for someone with A1 (beginner) language level.

        Words: {', '.join(unique_words)}

        Respond with ONLY the genuinely uncommon words separated by commas, nothing else. If there are no uncommon words, respond with "NONE".

        Be VERY selective - only include words that would be completely unfamiliar to beginners.
        Focus ONLY on words that are:
        1. Advanced vocabulary (B1 level or higher)
        2. Technical or specialized terms
        3. Abstract concepts difficult to explain simply
        4. Words with complex meanings that aren't part of everyday basic communication

        DO NOT include words that:
        1. Are part of basic A1 vocabulary
        2. Are common everyday words
        3. Have simple meanings that can be easily understood from context
        4. Are frequently used in beginner language courses

        Be strict and conservative in your selection - it's better to miss an uncommon word than to include a common one.
        """

        model = genai.GenerativeModel(
            model_name=config.WORD_TRANSLATION_MODEL,
            generation_config={
                "temperature": config.WORD_TRANSLATION_TEMPERATURE,
                "top_p": config.WORD_TRANSLATION_TOP_P,
                "top_k": config.WORD_TRANSLATION_TOP_K,
                "max_output_tokens": 100,
            },
            safety_settings=config.SAFETY_SETTINGS
        )

        response = model.generate_content(prompt)

        # Check if the response has candidates
        if hasattr(response, 'candidates') and response.candidates:
            if hasattr(response.candidates[0], 'content') and response.candidates[0].content:
                if hasattr(response.candidates[0].content, 'parts') and response.candidates[0].content.parts:
                    result = response.candidates[0].content.parts[0].text.strip()

                    if result.upper() == "NONE":
                        return []

                    # Parse the response and return the list of uncommon words
                    uncommon_words = [word.strip() for word in result.split(',')]
                    return uncommon_words

        # If we get here, there was an issue with the response structure
        # Check if there's prompt feedback indicating a block
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                logger.warning(f"Gemini blocked the uncommon words request: {response.prompt_feedback.block_reason}")

        # Fall back to a simple heuristic approach
        logger.info(f"Falling back to heuristic approach for identifying uncommon words in {language}")

        # Get the appropriate common words list based on language
        lang_key = language.lower()
        if lang_key in common_words:
            common_word_list = common_words[lang_key]
        else:
            # Default to English if language not in our list
            common_word_list = common_words["english"]

        # Identify uncommon words based on our simple list
        # Be more selective - only include words longer than 6 characters that aren't in our common list
        # This helps ensure we're only getting truly uncommon words
        uncommon_words = [word for word in unique_words if word.lower() not in common_word_list and len(word) > 6]

        # Limit to at most 2 words per sentence to avoid overwhelming translations
        return uncommon_words[:2]

    except Exception as e:
        logger.error(f"Error identifying uncommon words: {e}")

        # Fall back to a simple heuristic approach
        logger.info(f"Falling back to heuristic approach after exception for identifying uncommon words in {language}")

        # Get the appropriate common words list based on language
        lang_key = language.lower()
        if lang_key in common_words:
            common_word_list = common_words[lang_key]
        else:
            # Default to English if language not in our list
            common_word_list = common_words["english"]

        # Identify uncommon words based on our simple list
        # Be more selective - only include words longer than 6 characters that aren't in our common list
        # This helps ensure we're only getting truly uncommon words
        uncommon_words = [word for word in unique_words if word.lower() not in common_word_list and len(word) > 6]

        # Limit to at most 2 words per sentence to avoid overwhelming translations
        return uncommon_words[:2]

def translate_words(words: List[str], source_language: str) -> Dict[str, str]:
    """
    Translate a list of words from the source language to Turkish

    Args:
        words: List of words to translate
        source_language: The source language

    Returns:
        Dictionary mapping original words to their Turkish translations
    """
    if not words:
        return {}

    # Check cache first for all words
    translations = {}
    words_to_translate = []

    for word in words:
        cache_key = f"{word}_{source_language}_tr"
        if cache_key in word_translation_cache:
            translations[word] = word_translation_cache[cache_key]
        else:
            words_to_translate.append(word)

    if not words_to_translate:
        return translations

    # Fallback translations for truly uncommon words
    fallback_translations = {
        # German
        "aufmerksamkeit": "dikkat",
        "übermensch": "üstün insan",
        "philosophie": "felsefe",
        "wissenschaft": "bilim",
        "technologie": "teknoloji",
        "psychologie": "psikoloji",
        "mathematik": "matematik",
        "universität": "üniversite",
        "entwicklung": "gelişim",
        "gesellschaft": "toplum",
        "wirtschaft": "ekonomi",
        "regierung": "hükümet",
        "umgebung": "ortam",
        "erfahrung": "deneyim",

        # English
        "philosophy": "felsefe",
        "science": "bilim",
        "technology": "teknoloji",
        "psychology": "psikoloji",
        "mathematics": "matematik",
        "university": "üniversite",
        "development": "gelişim",
        "society": "toplum",
        "economy": "ekonomi",
        "government": "hükümet",
        "environment": "ortam",
        "experience": "deneyim",
        "quantum": "kuantum",
        "physics": "fizik",
        "thermodynamics": "termodinamik",

        # French
        "philosophie": "felsefe",
        "science": "bilim",
        "technologie": "teknoloji",
        "psychologie": "psikoloji",
        "mathématiques": "matematik",
        "université": "üniversite",
        "développement": "gelişim",
        "société": "toplum",
        "économie": "ekonomi",
        "gouvernement": "hükümet",
        "environnement": "ortam",
        "expérience": "deneyim",
        "astrophysique": "astrofizik",

        # Spanish
        "filosofía": "felsefe",
        "ciencia": "bilim",
        "tecnología": "teknoloji",
        "psicología": "psikoloji",
        "matemáticas": "matematik",
        "universidad": "üniversite",
        "desarrollo": "gelişim",
        "sociedad": "toplum",
        "economía": "ekonomi",
        "gobierno": "hükümet",
        "ambiente": "ortam",
        "experiencia": "deneyim",
        "arquitectura": "mimarlık",
        "neurociencia": "sinirbilim"
    }

    try:
        # Create a prompt to translate the words
        prompt = f"""
        Translate the following words from {source_language} to Turkish:

        {', '.join(words_to_translate)}

        Respond with ONLY the translations in the following format:
        original_word1: translation1
        original_word2: translation2
        ...

        Keep the translations simple and appropriate for A1 (beginner) level.
        """

        model = genai.GenerativeModel(
            model_name=config.WORD_TRANSLATION_MODEL,
            generation_config={
                "temperature": config.WORD_TRANSLATION_TEMPERATURE,
                "top_p": config.WORD_TRANSLATION_TOP_P,
                "top_k": config.WORD_TRANSLATION_TOP_K,
                "max_output_tokens": config.WORD_TRANSLATION_MAX_OUTPUT_TOKENS,
            },
            safety_settings=config.SAFETY_SETTINGS
        )

        response = model.generate_content(prompt)

        # Check if the response has candidates
        if hasattr(response, 'candidates') and response.candidates:
            if hasattr(response.candidates[0], 'content') and response.candidates[0].content:
                if hasattr(response.candidates[0].content, 'parts') and response.candidates[0].content.parts:
                    result = response.candidates[0].content.parts[0].text.strip()

                    # Parse the response
                    for line in result.split('\n'):
                        if ':' in line:
                            original, translation = line.split(':', 1)
                            original = original.strip()
                            translation = translation.strip()

                            # Add to translations dictionary
                            translations[original] = translation

                            # Cache the result
                            cache_key = f"{original}_{source_language}_tr"
                            word_translation_cache[cache_key] = translation

                    return translations

        # If we get here, there was an issue with the response structure
        # Check if there's prompt feedback indicating a block
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                logger.warning(f"Gemini blocked the translation request: {response.prompt_feedback.block_reason}")

        # Fall back to our dictionary for any words we can translate
        logger.info(f"Falling back to dictionary for translating words from {source_language}")
        for word in words_to_translate:
            word_lower = word.lower()
            if word_lower in fallback_translations:
                translations[word] = fallback_translations[word_lower]
                # Cache the result
                cache_key = f"{word}_{source_language}_tr"
                word_translation_cache[cache_key] = fallback_translations[word_lower]

        return translations

    except Exception as e:
        logger.error(f"Error translating words to Turkish: {e}")

        # Fall back to our dictionary for any words we can translate
        logger.info(f"Falling back to dictionary after exception for translating words from {source_language}")
        for word in words_to_translate:
            word_lower = word.lower()
            if word_lower in fallback_translations:
                translations[word] = fallback_translations[word_lower]
                # Cache the result
                cache_key = f"{word}_{source_language}_tr"
                word_translation_cache[cache_key] = fallback_translations[word_lower]

        return translations  # Return what we have so far

def process_text_with_translations(text: str, language: str) -> str:
    """
    Process text to add Turkish translations for truly uncommon words
    Only translates words that are genuinely difficult for beginners
    Places all translations at the very end of the response
    Removes any existing 'wort schatz' sections from the text

    Args:
        text: The text to process
        language: The language of the text

    Returns:
        Text with Turkish translations added for uncommon words only at the end
    """
    # Skip if the text is already in Turkish
    if language.lower() == "turkish":
        return text

    try:
        # Use a more comprehensive approach to clean the text
        # Split the text into lines
        lines = text.split('\n')
        cleaned_lines = []

        # Process each line
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip 'wort schatz' lines and their associated translations
            if line.lower() == 'wort schatz':
                # Skip this line and any following translation lines (format: word = translation)
                i += 1
                while i < len(lines) and '=' in lines[i]:
                    i += 1
                continue

            # Skip lines that end with 'wort schatz' and the following line if it contains '='
            if line.lower().endswith('wort schatz'):
                i += 1
                if i < len(lines) and '=' in lines[i]:
                    i += 1
                continue

            # Skip standalone translation lines (format: word = translation)
            if '=' in line and len(line.split('=')) == 2:
                word_part = line.split('=')[0].strip().lower()
                # Check if this looks like a translation line
                if len(word_part) > 0 and not line.startswith('*') and not line.startswith('-'):
                    i += 1
                    continue

            # Add the line to our cleaned lines
            cleaned_lines.append(line)
            i += 1

        # Rejoin the cleaned lines
        cleaned_text = '\n'.join(cleaned_lines)

        # Split text into sentences for processing
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)

        # Collect all translations
        all_translations = {}

        # Process each sentence to find uncommon words and their translations
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue

            try:
                # Identify uncommon words in the sentence
                uncommon_words = identify_uncommon_words(sentence, language)

                # If no uncommon words, continue to next sentence
                if not uncommon_words:
                    continue

                # Translate uncommon words to Turkish
                translations = translate_words(uncommon_words, language)

                # If no translations were found, continue to next sentence
                if not translations:
                    continue

                # Add translations to the collection
                for word, translation in translations.items():
                    # Only include translations for words that were identified as uncommon
                    if word.lower() in [w.lower() for w in uncommon_words]:
                        all_translations[word] = translation

            except Exception as sentence_error:
                logger.error(f"Error processing sentence for translations: {sentence_error}")
                continue  # Continue to the next sentence on error

        # If no translations were found, return the cleaned text
        if not all_translations:
            return cleaned_text

        # Format the final response with all translations at the end
        result = cleaned_text

        # Add a section for all translations at the end
        result += "\n\nwort schatz"
        for word, translation in all_translations.items():
            result += f"\n{word} = {translation}"

        # Add emoji to the last translation for a friendly touch
        if all_translations:
            result += " 😊"

        return result
    except Exception as e:
        logger.error(f"Error in process_text_with_translations: {e}")
        return text  # Return the original text on error

def post_process_response(response: str, language: str) -> str:
    """
    Post-process the bot's response to add Turkish translations for all non-Turkish languages

    Args:
        response: The bot's response
        language: The detected language

    Returns:
        Processed response with translations
    """
    try:
        # Skip if the response is already in Turkish
        if language.lower() == "turkish":
            return response

        # Skip if the response is empty or None
        if not response:
            logger.warning("Received empty response for translation processing")
            return response

        # Process all non-Turkish languages to add translations
        logger.info(f"Adding Turkish translations for uncommon words in {language} response")
        processed_response = process_text_with_translations(response, language)

        # If processing failed and returned None, return the original response
        if processed_response is None:
            logger.warning("Translation processing returned None, using original response")
            return response

        return processed_response
    except Exception as e:
        logger.error(f"Error in post_process_response: {e}")
        return response  # Return the original response on error
